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
from selenium_helper import init_selenium_driver
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
        temperature=0.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# Parsing a playground page for preprocessing steps -> NLP and parsing the code layers (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

#  List of 147 Playground slugs (or however you get your list)
def parse_playground_kaggle(max_competitions: int = 5) -> list[str]:
    """
    Scrape Kaggle’s Playground filter pages (1–8) and return up to max_competitions slugs.
    """
    slugs: list[str] = []
    driver = init_selenium_driver()

    pattern = re.compile(r"^/competitions/([A-Za-z0-9].*)$")
    for page_num in range(1, 9):
        url = f"https://www.kaggle.com/competitions?hostSegmentIdFilter=8&page={page_num}"
        driver.get(url)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href^="/competitions/"]'))
            )
        except:
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.select('a[href^="/competitions/"]'):
            href = a["href"]
            m = pattern.match(href)
            if not m:
                continue
            slug = m.group(1).split("?")[0]
            if slug not in slugs:
                slugs.append(slug)
                if len(slugs) >= max_competitions:
                    driver.quit()
                    return slugs

    driver.quit()
    return slugs


def parse_playground_kaggle_from(start_slug: str, max_competitions: int = 5) -> list[str]:
    """
    Scrape Kaggle’s Playground filter pages (1–8) and return up to max_competitions slugs
    starting with the given start_slug as the first entry. If start_slug is not found,
    it collects from the very beginning.
    """
    slugs: list[str] = []
    driver = init_selenium_driver()
    pattern = re.compile(r"^/competitions/([A-Za-z0-9].*)$")
    started = start_slug is None

    for page_num in range(1, 9):
        url = f"https://www.kaggle.com/competitions?hostSegmentIdFilter=8&page={page_num}"
        driver.get(url)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href^="/competitions/"]'))
            )
        except:
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.select('a[href^="/competitions/"]'):
            href = a["href"]
            m = pattern.match(href)
            if not m:
                continue
            slug = m.group(1).split("?")[0]

            if not started:
                if slug == start_slug:
                    started = True
                    slugs.append(slug)
                else:
                    continue
            else:
                if slug not in slugs:
                    slugs.append(slug)

            if len(slugs) >= max_competitions:
                driver.quit()
                return slugs

    driver.quit()
    return slugs



# ─────────────────────────────────────────────────────────────────────────────
# Ask LLM for structured description and dataset description (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm_for_structured_output(comp_meta: str, notebook_text: str) -> dict:
    # 1) System prompt
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

    
    # 2) First user message: raw JSON payload
    payload = {
        "competition_metadata": comp_meta["competition_metadata"],
        "dataset_metadata": comp_meta["dataset_metadata"],
        "notebook_text": notebook_text
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    # 3) Second user message: output‐format instructions
    user_instructions = (
            "Now produce EXACTLY the JSON described by the function schema—no extras, no markdown fences. "
        )    


    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
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

    # 1) Try strict JSON
    for loader in (json.loads, lambda s: json.loads(_trim_to_braces(s))):
        try:
            args = loader(raw)
            break
        except Exception:
            args = None
    else:
        # 2) Try Python literal
        try:
            args = ast.literal_eval(_trim_to_braces(raw))
        except Exception as e:
            print(f"[WARN] Couldn’t parse LLM output at all: {e}")
            return None

    # 3) Validate required keys
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


# ─────────────────────────────────────────────────────────────────────────────
# Collect Top‐Voted DL Notebooks (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

def collect_and_structured(max_per_keyword: int = 5, start: str = None) -> pd.DataFrame:
    # Determine CSV mode and write header only when starting fresh
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

    # # Fetch slugs, either fresh or resuming
    # if start:
    #     all_slugs = parse_playground_kaggle_from(start, num_competitions)
    # else:
    #     all_slugs = parse_playground_kaggle(num_competitions)


    records = []
    driver = init_selenium_driver()

    wait = 0 if start == None else 1
    for slug in train:
        if slug == start: wait = 0
        if wait: continue
        print(f"\n[INFO] Processing competition: {slug}")
        comp_folder = Path("train") / slug
        comp_folder.mkdir(parents=True, exist_ok=True)

        # — 1) Scrape & parse HTML →
        html      = fetch_competition_page_html(slug, driver)
        comp_meta = parse_competition_metadata(html)
        comp_meta["slug"] = slug

        # now fetch & parse the /data tab
        data_html = fetch_competition_page_html(f"{slug}/data", driver)
        temp = parse_competition_data_tab(data_html)  
        comp_meta["dataset_metadata"] =  temp["dataset_metadata"]
        comp_meta["files_list"] = temp["files_list"]

        labels = label_competition(comp_meta)
        comp_meta.update(labels)
        print(comp_meta["target_column"])
        print(comp_meta["training_files"])
        print(comp_meta["evaluation_metrics"])        

        
        ##In case we need to download them
        # downloaded_paths = download_train_file(
        #     comp_meta["slug"],
        #     comp_folder,
        #     comp_meta["training_files"]
        # )

        # # 3) profile each one
        # all_schemas = {}
        # for p in downloaded_paths:
        #     tabular = extract_tabular(str(p))
        #     schema = describe_schema(
        #         source_path=tabular,
        #         target_column=comp_meta["target_column"]
        #     )
        #     all_schemas[p.name] = schema

        # # now you have a dict mapping each filename → its profile
        # comp_meta["data_profiles"] = all_schemas
        # print(comp_meta["data_profiles"])

        py_exporter = PythonExporter()


        tf_count, pt_count = 0, 0


        # ───── (A) List TensorFlow‐tagged kernels ─────
        try:
            proc_tf = subprocess.run(
                [
                    "kaggle", "kernels", "list",
                    "--competition", slug,
                    "-s", "tensorflow",
                    "--sort-by", "voteCount",
                    "--page-size", "50",
                    "-v",  # CSV output
                ],
                capture_output=True, text=True, check=True
            )
            df_tf = pd.read_csv(pd.io.common.StringIO(proc_tf.stdout))
        except Exception as e:
            print(f"[WARN] TF list failed for {slug}: {e}")
            df_tf = pd.DataFrame()

        # ───── (B) Iterate TF candidates: download → LLM → keep/drop ─────
        for _, row in df_tf.iterrows():

            if tf_count >= max_per_keyword:
                break

            kernel_ref = row["ref"]  # e.g. "username/kernel‐slug"
            username, kernel_name = kernel_ref.split("/", 1)
            kernel_link = f"https://www.kaggle.com/{kernel_ref}"

            # Possible local filenames
            ipynb_path = comp_folder / f"{kernel_name}.ipynb"
            py_path    = comp_folder / f"{kernel_name}.py"

            # Download if neither exists
            if not (ipynb_path.exists() or py_path.exists()):
                try:
                    subprocess.run(
                        ["kaggle", "kernels", "pull", kernel_ref, "-p", str(comp_folder)],
                        check=True
                    )
                except Exception as e:
                    print(f"   [WARN] Failed to pull {kernel_ref}: {e}")
                    continue

            # Choose whichever file actually exists
            if ipynb_path.exists():
                final_path = ipynb_path
                lang = "ipynb"
            elif py_path.exists():
                final_path = py_path
                lang = "py"
            else:
                print(f"   [WARN] Neither {ipynb_path} nor {py_path} found for {kernel_ref}. Skipping.")
                continue

            # Read the notebook (full text)
            try:
                if lang == "ipynb":
                    # load the notebook
                    with open(final_path, "r", encoding="utf-8") as f:
                        nb_node = nb_read(f, as_version=4)
                    # export to .py text
                    text_content, _ = py_exporter.from_notebook_node(nb_node)
                else:
                    # just read the .py file
                    with open(final_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
            except Exception as e:
                print(f"   [WARN] Failed to extract text from {final_path}: {e}")
                text_content = ""

            # Immediately ask the LLM for structured output
            struct = ask_llm_for_structured_output(comp_meta, text_content)


            if struct is None:
                continue

            # Only keep if LLM says “used_technique” == “DL” AND library == “TensorFlow”
            if struct.get("used_technique", "").upper() == "DL" and \
            struct.get("library", "").lower() == "tensorflow":
            
                print("TARGET-TF: " + str(struct["target_column"]))

                # Write that ten‐field JSON into our CSV (only DL entries)
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
                rec = {
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
                }
                records.append(rec)
                tf_count += 1
                print(f"   [KEPT→TF] {kernel_ref}  (votes={row.get('totalVotes', 0)})")

        # ───── (C) Now do the same for PyTorch‐tagged kernels ─────
        try:
            proc_pt = subprocess.run(
                [
                    "kaggle", "kernels", "list",
                    "--competition", slug,
                    "-s", "pytorch",
                    "--sort-by", "voteCount",
                    "--page-size", "50",
                    "-v",  # CSV output
                ],
                capture_output=True, text=True, check=True
            )
            df_pt = pd.read_csv(pd.io.common.StringIO(proc_pt.stdout))
        except Exception as e:
            print(f"[WARN] PT list failed for {slug}: {e}")
            df_pt = pd.DataFrame()

        for _, row in df_pt.iterrows():
            if tf_count == 1: break
            if pt_count >= max_per_keyword:
                break

            kernel_ref = row["ref"]
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
                    print(f"   [WARN] Failed to pull {kernel_ref}: {e}")
                    continue

            if ipynb_path.exists():
                final_path = ipynb_path
                lang = "ipynb"
            elif py_path.exists():
                final_path = py_path
                lang = "py"
            else:
                print(f"   [WARN] Neither {ipynb_path} nor {py_path} found for {kernel_ref}. Skipping.")
                continue

            # Read the notebook’s contents
            try:
                if lang == "ipynb":
                    # load the notebook
                    with open(final_path, "r", encoding="utf-8") as f:
                        nb_node = nb_read(f, as_version=4)
                    # export to .py text
                    text_content, _ = py_exporter.from_notebook_node(nb_node)
                else:
                    # just read the .py file
                    with open(final_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
            except Exception as e:
                print(f"   [WARN] Failed to extract text from {final_path}: {e}")
                text_content = ""

                
            struct = ask_llm_for_structured_output(comp_meta, text_content)
            
            if struct is None:
                continue

            # Only keep if LLM says “used_technique” == “DL” AND library == “PyTorch”
            if struct.get("used_technique", "").upper() == "DL" and \
            struct.get("library", "").lower() == "pytorch":
        
                print("TARGET-PYT: " + str(struct["target_column"]))

                # Write that ten‐field JSON into our CSV
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

                rec = {
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
                    "library":                         struct["library"],          # "PyTorch"
                    "kernel_ref":                      kernel_ref,
                    "kernel_link":                     kernel_link,
                    #"training_files":                  struct["training_files"],
                    "username":                        username,
                    "last_run_date":                   row.get("lastRunTime"),
                    "votes":                           row.get("totalVotes"),
                    "downloaded_path":                 str(final_path),
                    "language":                        lang,
                    "dl_keyword":                      "pytorch"
                }
                records.append(rec)
                pt_count += 1
                print(f"   [KEPT→PT] {kernel_ref}  (votes={row.get('totalVotes', 0)})")

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
