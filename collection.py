import json
import re
import csv
import subprocess
import time
import pickle
from pathlib import Path
import io

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import openai
from selenium_helper import init_selenium_driver
from config import OPENAI_MODEL,EXCEL_FILE,MAX_NOTEBOOK_TOKENS,ENCODER,kaggle_api
from utils import fetch_competition_page_html, parse_competition_metadata, parse_competition_data_tab,describe_schema, download_train_file, extract_tabular


#Get the target column
def label_competition(comp_meta: dict) -> dict:
    """
    Fill in comp_meta with:
      - target_column: list of all label columns in train.csv
    Sends both competition_metadata and dataset_metadata to the LLM.
    """
    system = {
        "role": "system",
        "content": (
            "You are an expert data scientist.  "
            "From the competition and dataset metadata, emit **only** a JSON object with exactly these keys:\n"
            "target_column: an array of all column names in train.csv that must be predicted\n"
            "training_files: Based on dataset_metadata give [`<string>`, …],  an array of all training tabular files that need to be downloaded\n"
            "No extra keys, no prose, no markdown—just a JSON object."
            "You also have access to `dataset_metadata` in the competition_metadata payload—use them to ground your explanations.\n"
        )
    }

    payload = {
        "competition_metadata": comp_meta["competition_metadata"],
        "dataset_metadata": comp_meta["dataset_metadata"]
    }

    user = {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False)
    }

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system, user]
    )
    out = resp.choices[0].message.content.strip()
    if out.startswith("```"):
        out = "\n".join(out.split("\n")[1:-1]).strip()
    result = json.loads(out)

    # Normalize: ensure the key is 'target_column' holding a list
    if "target_columns" in result and "target_column" not in result:
        result["target_column"] = result.pop("target_columns")
    elif "target_column" in result and not isinstance(result["target_column"], list):
        # wrap single string in a list
        result["target_column"] = [result["target_column"]]

    # Normalize: ensure the key 'training_files' holding a list
    if "training_file" in result and "training_files" not in result:
        result["training_files"] = [result.pop("training_file")]
    elif "training_files" in result and not isinstance(result["training_files"], list):
        result["training_files"] = [result["training_files"]]
    return result



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

def ask_llm_for_structured_output(comp_meta: str, notebook_path: str) -> dict:
    # 1) System prompt
    system_prompt = (
        "You are an expert data scientist. "
        "**Under no circumstances should you reference, draw from, or quote any Kaggle machine-learning notebooks, examples, code snippets or commentary.** "
        "**Do not use or identify any of the following traditional ML methods or their variants/abbreviations in your analysis**: "
        "Linear Regression (LR), Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Extra Trees (ET), "
        "AdaBoost, Gradient Boosting Machine (GBM), XGBoost (XGB), LightGBM (LGBM), CatBoost (CB), Support Vector Machine (SVM), "
        "k-Nearest Neighbors (KNN), Naive Bayes (NB), Principal Component Analysis (PCA), SMOTE, feature selection, "
        "ensemble learning, tree-based models, boosting, bagging."
        "Also extract the exact name of the target columns (labels) and return them as \"target_column\""
    )      

    # Read the raw notebook file to pass as context (we truncate if too long)
    with open(notebook_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Truncate notebook text if it exceeds token budget
    tokens = ENCODER.encode(raw)
    if len(tokens) <= MAX_NOTEBOOK_TOKENS:
        text_trunc = raw
    else:
        text_trunc = ENCODER.decode(tokens[:MAX_NOTEBOOK_TOKENS])


    # 2) First user message: raw JSON payload
    payload = {
        "competition_metadata": comp_meta["competition_metadata"],
        "dataset_metadata": comp_meta["dataset_metadata"],
        "notebook_text": text_trunc
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    # 3) Second user message: output‐format instructions
    user_instructions = (
        "Please respond with a JSON object exactly with these keys (no extra keys!):\n"
        "{\n"
        '  "competition_problem_type": "classification|regression",\n'
        '  "competition_problem_subtype": (a single, concise phrase—lower-case words and hyphens only—describing the specific subtype, e.g. “binary classification”, “multiclass classification”, “multi-label classification”, “time-series forecasting”, “continuous regression”, “ordinal regression”, etc. or any other that fits.)\n"'
        '  "competition_problem_description": "Dense, short, and detailed description of the problem, what needs to be found, no repetitive words, do not include the dataset description here",\n'
        '  "dataset_metadata": "Rewrite the given dataset_metadata in plain English as a single coherent paragraph, removing any non-human symbols (no bullets, special characters, or markdown)."\n'
        '  "competition_dataset_type": "choose exactly one primary data modality from this list (capitalized exactly): Tabular, Time-series, Text, Image, Audio, Video, Geospatial, Graph, Multimodal. ",\n'
        '  "preprocessing_steps": [\n'
        '      "<step 1 in plain English>",\n'
        '      "<step 2>",\n'
        '      "…"\n'
        '  ],\n'
        '  "notebook_model_layers_code": "<complete code snippet defining each layer with all its parameters>",\n'
        '  "used_technique": "<\\"DL\\" or \\"ML\\">",\n'
        '  "library": "<library>",\n'
        '  "target_column": [ "<string>", … ] an array of all column names in train.csv that must be predicted \n'
        "}\n\n"
        "You also have access to `dataset_metadata` in the competition_metadata payload—use them to ground your explanations.\n"
        "- For `preprocessing_steps`, list every transformation (scaling, normalization, one-hot encoding, etc.) in plain English.\n"
        "- For `notebook_model_layers_code`, include the literal code lines of model compile, model fit, and that build each layer (e.g. `Dense(128, activation='relu', …)`).\n"
        "- Keep everything extremely dense and factual—no extra keys, no markdown fences, just valid JSON."
    )

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_payload},
        {"role": "user",    "content": user_instructions},
    ]

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=messages
    )

    content = response.choices[0].message.content.strip()
    
    # strip triple‐backticks if any
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1]).strip()
        
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("[ERROR] LLM did not return valid JSON. Raw response:")
        print(content)
        return None
    





# ─────────────────────────────────────────────────────────────────────────────
# Collect Top‐Voted DL Notebooks (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

def collect_and_structured(num_competitions: int = 10, max_per_keyword: int = 5, start: str = None) -> pd.DataFrame:
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
            "dataset_metadata",
            "target_column", 
            "preprocessing_steps",
            "notebook_model_layers_code",
            "used_technique",
            "library",
            "kernel_ref",
            "kernel_link"
            "training_files"
        ])

    # Fetch slugs, either fresh or resuming
    if start:
        all_slugs = parse_playground_kaggle_from(start, num_competitions)
    else:
        all_slugs = parse_playground_kaggle(num_competitions)


    records = []
    driver = init_selenium_driver()
    t = 0
    for slug in all_slugs:
        if t == 2: break
        t+=1
        print(f"\n[INFO] Processing competition: {slug}")
        comp_folder = Path("solutions") / slug
        comp_folder.mkdir(parents=True, exist_ok=True)

        # — 1) Scrape & parse HTML →
        html      = fetch_competition_page_html(slug, driver)
        comp_meta = parse_competition_metadata(html)
        comp_meta["slug"] = slug

        # now fetch & parse the /data tab
        data_html = fetch_competition_page_html(f"{slug}/data", driver)
        comp_meta["dataset_metadata"] = parse_competition_data_tab(data_html)   

        labels = label_competition(comp_meta)
        comp_meta.update(labels)
        print(comp_meta["target_column"])
        print(comp_meta["training_files"])
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
                if final_path.suffix == ".ipynb":
                    with open(final_path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                    text_content = ""
                    for cell in nb.get("cells", []):
                        text_content += " ".join(cell.get("source", [])) + " "
                else:
                    with open(final_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
            except Exception:
                text_content = ""

            # Immediately ask the LLM for structured output
            struct = ask_llm_for_structured_output(comp_meta, str(final_path))


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
                    "dataset_description":             struct["dataset_metadata"],
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
                if final_path.suffix == ".ipynb":
                    with open(final_path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                    text_content = ""
                    for cell in nb.get("cells", []):
                        text_content += " ".join(cell.get("source", [])) + " "
                else:
                    with open(final_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
            except Exception:
                text_content = ""

            struct = ask_llm_for_structured_output(comp_meta, str(final_path))
            
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
                    "competition_dataset_description": struct["dataset_metadata"],
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
