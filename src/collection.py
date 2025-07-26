import json
import re
import csv
import subprocess
import time
import pickle
from pathlib import Path
import io, ast

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from nbformat import read as nb_read
from nbconvert import PythonExporter

import openai
from helpers.selenium_helper import init_selenium_driver
from config import OPENAI_MODEL,EXCEL_FILE
from utils import fetch_competition_page_html, parse_competition_metadata, parse_competition_data_tab,describe_schema, extract_tabular, download_train_file
from prompts import label_competition_schema, ask_structured_schema
from comps import train

#Get the target column
def label_competition(comp_meta: dict) -> dict:

    
    # build our messages
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert data scientist.  "
            "Use the provided competition_metadata and dataset_metadata to fill exactly two fields:\n"
            "  1) target_column: an array of all column names in the dataset that must be predicted\n"
            "  2) training_files: Based on dataset_metadata give [<string>, …],  an array of all training tabular files that need to be downloaded\n"
            "  3) evaluation_metrics: based on the competition_metadata, retrieve the evaluation metrics used in the competition"
            "Emit ONLY those two keys as JSON—no extra keys, no prose, no markdown."
        )
    }
    user_msg = {
        "role": "user",
        "content": json.dumps({
            "competition_metadata": comp_meta["competition_metadata"],
            "dataset_metadata":   comp_meta["dataset_metadata"]
        }, ensure_ascii=False)
    }

    # call the model with our function schema
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[system_msg, user_msg],
        functions=[label_competition_schema],
        function_call={"name": "label_competition_schema"}  # force this function
    )

    # parse out the function call arguments
    content = response.choices[0].message

    if content.function_call is None:
        raise RuntimeError("Model did not call label_competition_schema")


    args = json.loads(content.function_call.arguments)
    target_cols    = args.get("target_column", [])
    training_files = args.get("training_files", [])
    evaluation_metrics =  args.get("evaluation_metrics", [])

    # normalize to lists
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    if isinstance(training_files, str):
        training_files = [training_files]

    return {
        "target_column":  target_cols,
        "training_files": training_files,
        "evaluation_metrics": evaluation_metrics
    }



def get_comp_files(slug: str):

    proc = subprocess.run(
        ["kaggle", "competitions", "files", slug, "-v", "-q"],
        capture_output=True, text=True, check=True
    )
    reader = csv.reader(io.StringIO(proc.stdout))
    next(reader, None)  
    for row in reader:
        print(row[0])



# Ask LLM for structured description and dataset description (tensorflow & pytorch)
def ask_llm_for_structured_output(comp_meta: str, notebook_text: str) -> dict:
    system_prompt = (
        "You are an expert data scientist. "
        "***Provide a dense and factual description of the competition_description, full dataset_metadata, exact problem type, subtype\n"
        "***Based on the notebook, provide the preprocessing steps as a list of string used in the notebooks, along with the code snippets of layers, compile, and fit \n" 
        "**Under no circumstances should you reference, draw from, or quote any Kaggle machine-learning notebooks, examples, code snippets or commentary.** "
        "**Do not use or identify any of the following traditional ML methods or their variants/abbreviations in your analysis**: "
        "Linear Regression (LR), Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Extra Trees (ET), "
        "AdaBoost, Gradient Boosting Machine (GBM), XGBoost (XGB), LightGBM (LGBM), CatBoost (CB), Support Vector Machine (SVM), "
        "k-Nearest Neighbors (KNN), Naive Bayes (NB), Principal Component Analysis (PCA), SMOTE, feature selection, "
        "ensemble learning, tree-based models, boosting, bagging."
    )      

    
    payload = {
        "competition_metadata": comp_meta["competition_metadata"],
        "dataset_metadata": comp_meta["dataset_metadata"],
        "notebook_text": notebook_text
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    user_instructions = (
            "Now produce EXACTLY the JSON described by the function schema—no extras, no markdown fences. "
        )    


    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages= [
            {"role":"system", **{"content": system_prompt}},
            {"role": "user",    "content": user_payload},
            {"role": "user",    "content": user_instructions},
        ],
        functions=[ask_structured_schema],
        function_call={"name": "ask_structured_schema"}
    )

    content = response.choices[0].message
    if not content.function_call:
        raise RuntimeError("LLM did not call ask_structured_schema()")

    content = response.choices[0].message
    raw = content.function_call and content.function_call.arguments
    if not raw:
        print("[WARN] No function_call.arguments at all")
        return None

    for loader in (json.loads, lambda s: json.loads(_trim_to_braces(s))):
        try:
            args = loader(raw)
            break
        except Exception:
            args = None
    else:
        try:
            args = ast.literal_eval(_trim_to_braces(raw))
        except Exception as e:
            print(f"[WARN] Couldn’t parse LLM output at all: {e}")
            return None

    required = ask_structured_schema["parameters"]["required"]
    if not all(k in args for k in required):
        missing = [k for k in required if k not in args]
        print(f"[WARN] Parsed JSON missing keys: {missing}")
        return None

    return args

def _trim_to_braces(s: str) -> str:
    """Extract the substring from the first { to the last }."""
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if i != -1 and j != -1 else s



def collect_tagged_kernels(
    slug: str,
    tag: str,                      # "tensorflow"  /  "pytorch"
    max_keep: int,
    comp_folder: Path,
    comp_meta: dict,
    csv_writer: csv.writer,
    records: list,
    py_exporter: PythonExporter,
) -> int:

    kept = 0


    try:
        proc = subprocess.run(
            ["kaggle","kernels","list",
             "--competition", slug,
             "-s", tag,
             "--sort-by","voteCount",
             "--page-size","50",
             "-v"],
            capture_output=True, text=True, check=True
        )
        df = pd.read_csv(pd.io.common.StringIO(proc.stdout))
    except Exception as e:
        print(f"[WARN] {tag} list failed for {slug}: {e}")
        return 0


    for _, row in df.iterrows():
        if kept >= max_keep:
            break

        kernel_ref = row["ref"]                    # "user/kernel"
        username, kernel_name = kernel_ref.split("/", 1)
        kernel_link = f"https://www.kaggle.com/{kernel_ref}"

        ipynb_path = comp_folder / f"{kernel_name}.ipynb"
        py_path    = comp_folder / f"{kernel_name}.py"


        if not (ipynb_path.exists() or py_path.exists()):
            try:
                subprocess.run(
                    ["kaggle", "kernels", "pull", kernel_ref, "-p", str(comp_folder)],
                    check=True
                )
            except Exception as e:
                print(f"   [WARN] pull failed {kernel_ref}: {e}")
                continue


        if ipynb_path.exists():
            final_path, lang = ipynb_path, "ipynb"
            try:
                with open(final_path, "r", encoding="utf-8") as f:
                    nb_node = nb_read(f, as_version=4)
                text_content, _ = py_exporter.from_notebook_node(nb_node)
            except Exception as e:
                print(f"   [WARN] nb read {kernel_ref}: {e}")
                continue
        elif py_path.exists():
            final_path, lang = py_path, "py"
            try:
                text_content = py_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"   [WARN] py read {kernel_ref}: {e}")
                continue
        else:
            print(f"   [WARN] no file for {kernel_ref}")
            continue


        struct = ask_llm_for_structured_output(comp_meta, text_content)
        if not struct:
            continue

        if struct.get("used_technique","").upper()!="DL" \
           or struct.get("library","").lower()!=tag:
            continue 

        csv_writer.writerow([
            comp_meta["slug"],
            struct["competition_problem_type"],
            struct["competition_problem_subtype"],
            struct["competition_problem_description"],
            struct["competition_dataset_type"],
            comp_meta["evaluation_metrics"],
            struct["dataset_metadata"],
            struct["target_column"], 
            json.dumps(struct["preprocessing_steps"], ensure_ascii=False),
            struct["notebook_model_layers_code"],
            struct["used_technique"],
            struct["library"],
            kernel_ref, 
            kernel_link,
            #struct["training_files"]
        ])

        # Build the combined record
        records.append({
            "competition_slug":                comp_meta["slug"],
            "competition_problem_type":        struct["competition_problem_type"],
            "competition_problem_subtype":     struct["competition_problem_subtype"],
            "competition_problem_description": struct["competition_problem_description"],
            "competition_dataset_type":        struct["competition_dataset_type"],
            "evaluation_metrics":              comp_meta["evaluation_metrics"],
            "dataset_metadata":                struct["dataset_metadata"],
            "target_column":                   struct["target_column"],
            "preprocessing_steps":             struct["preprocessing_steps"],
            "notebook_model_layers_code":      struct["notebook_model_layers_code"],
            "used_technique":                  struct["used_technique"],   # "DL"
            "library":                         struct["library"],          # "TensorFlow"
            "kernel_ref":                      kernel_ref,
            "kernel_link":                     kernel_link,
            #"training_files":                  struct["training_files"],
            "username":                        username,
            "last_run_date":                   row.get("lastRunTime"),
            "votes":                           row.get("totalVotes"),
            "downloaded_path":                 str(final_path),
            "language":                        lang,
            "dl_keyword":                      "tensorflow"
        })

        kept += 1
        print(f"   [KEPT→{tag.upper()}] {kernel_ref} (votes={row.get('totalVotes',0)})")

    return kept



# Collect Top‐Voted DL Notebooks (tensorflow & pytorch)
def collect_and_structured(max_per_keyword: int = 5, start: str = None) -> pd.DataFrame:
    csv_mode = "w" if start is None else "a"
    write_header = start is None or not Path("notebooks_structured.csv").exists()

    csv_file = open("notebooks_structured.csv", csv_mode, encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "competition_slug",
            "competition_problem_type",
            "competition_problem_subtype",
            "competition_problem_description",
            "competition_dataset_type",
            "evaluation_metrics",
            "dataset_metadata",
            "target_column", 
            "preprocessing_steps",
            "notebook_model_layers_code",
            "used_technique",
            "library",
            "kernel_ref",
            "kernel_link",
        ])

    records = []
    driver = init_selenium_driver()

    wait = 0 if start == None else 1
    for slug in train:
        if slug == start: wait = 0
        if wait: continue
        print(f"\n[INFO] Processing competition: {slug}")
        comp_folder = Path("train") / slug
        comp_folder.mkdir(parents=True, exist_ok=True)

        html      = fetch_competition_page_html(slug, driver)
        comp_meta = parse_competition_metadata(html)
        comp_meta["slug"] = slug

        data_html = fetch_competition_page_html(f"{slug}/data", driver)
        temp = parse_competition_data_tab(data_html)  
        comp_meta["dataset_metadata"] =  temp["dataset_metadata"]
        comp_meta["files_list"] = temp["files_list"]

        labels = label_competition(comp_meta)
        comp_meta.update(labels)


        print(comp_meta["target_column"])
        print(comp_meta["evaluation_metrics"])        

        


        py_exporter = PythonExporter()

        tf_count, pt_count = 0, 0

        tf_count = collect_tagged_kernels(slug, "tensorflow", max_per_keyword,
                                    comp_folder, comp_meta,
                                    csv_writer, records, py_exporter)

        pt_count = collect_tagged_kernels(slug, "pytorch", max_per_keyword,
                                    comp_folder, comp_meta,
                                    csv_writer, records, py_exporter)

        print(f"  → Kept {tf_count} TensorFlow‐DL and {pt_count} PyTorch‐DL notebooks for {slug}")

    driver.quit()
    csv_file.close()

    # Build a DataFrame of only the notebooks that the LLM flagged as real DL
    df_structured = pd.DataFrame(records)

    # Save that DataFrame to Excel (optional)
    df_structured.to_excel(EXCEL_FILE, index=False)
    print(f"[INFO] Structured data saved to {EXCEL_FILE}")
    print(df_structured)
    return df_structured
