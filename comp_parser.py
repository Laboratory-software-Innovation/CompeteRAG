import json
import re
import csv
import subprocess
import time
import pickle
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import openai
from selenium_helper import init_selenium_driver
from config import OPENAI_MODEL,EXCEL_FILE,MAX_NOTEBOOK_TOKENS,ENCODER,kaggle_api
from utils import fetch_competition_page_html, parse_competition_metadata,describe_schema


#Get the target column
def label_competition(comp_meta: dict) -> dict:
    """
    Fill in comp_meta with:
      - target_column

    """
    system = {
      "role": "system",
      "content": (
        "You are an expert data scientist.  "
        "From the competition metadata, emit **only** a JSON object with exactly these keys:\n"
        "  • target_column (exact name of the label column in train.csv)\n"
        "  • competition_problem_type (must be “classification” or “regression”)\n"
        "  • competition_problem_subtype (a single concise lowercase hyphenated phrase describing the specific subtype, e.g. “binary-classification”, “multiclass-classification”, “time-series-forecasting”, etc.)\n"
        "No extra keys, no prose, no markdown—just a JSON object."
      )
    }
    user = {
      "role": "user",
      "content": json.dumps({"competition_metadata": comp_meta}, ensure_ascii=False)
    }
    resp = openai.chat.completions.create(
      model=OPENAI_MODEL, temperature=0.0, messages=[system, user]
    )
    out = resp.choices[0].message.content.strip()
    if out.startswith("```"):
      out = "\n".join(out.split("\n")[1:-1]).strip()
    return json.loads(out)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing a playground page for preprocessing steps -> NLP and parsing the code layers (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

#  List of 147 Playground slugs (or however you get your list)
def parse_playground_kaggle(max_competitions: int = 108) -> list[str]:
    """
    Scrape Kaggle’s Playground filter pages (1–8) and return up to max_competitions slugs.
    """

    non = ["playground-series-s3e23",
        "playground-series-s3e20",
        "playground-series-s3e18",
        "playground-series-s3e17",
        "playground-series-s3e12",
        "playground-series-s3e10",
        "playground-series-s3e8",
        "playground-series-s3e6",
        "playground-series-s3e4",
        "playground-series-s3e2",
        "scrabble-player-rating",
        "bigquery-geotab-intersection-congestion",
        "dont-call-me-turkey",
        "transfer-learning-on-stack-exchange-tags",
        "pubg-finish-placement-prediction",
        "new-york-city-taxi-fare-prediction",
        "costa-rican-household-poverty-prediction",
        "santas-uncertain-bags",
        "kobe-bryant-shot-selection",
        "finding-elo"]
    
    slugs: list[str] = []
    driver = init_selenium_driver()

    pattern = re.compile(r"^/competitions/([A-Za-z0-9].*)$")
    for page_num in range(1, 6):
        url = f"https://www.kaggle.com/competitions?tagIds=14101&hostSegmentIdFilter=8&page={page_num}"
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
            if slug not in (slugs and non):
                slugs.append(slug)
                if len(slugs) >= max_competitions:
                    driver.quit()
                    return slugs

    driver.quit()
    return slugs

names = parse_playground_kaggle(108)
for i in names: 
    print("\""+i+"\"")

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
        url = f"https://www.kaggle.com/competitions?tagIds=14101&hostSegmentIdFilter=8&page={page_num}"
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

def ask_llm_for_structured_output(comp_meta: dict, notebook_path: str) -> dict:
    # 1) System prompt
    system_prompt = (
        "You are an expert data scientist. "
        "**Under no circumstances should you reference, draw from, or quote any Kaggle machine-learning notebooks, examples, code snippets or commentary.** "
        "**Do not use or identify any of the following traditional ML methods or their variants/abbreviations in your analysis**: "
        "Linear Regression (LR), Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Extra Trees (ET), "
        "AdaBoost, Gradient Boosting Machine (GBM), XGBoost (XGB), LightGBM (LGBM), CatBoost (CB), Support Vector Machine (SVM), "
        "k-Nearest Neighbors (KNN), Naive Bayes (NB), Principal Component Analysis (PCA), SMOTE, feature selection, "
        "ensemble learning, tree-based models, boosting, bagging."
        "Also extract the exact name of the target column (label) and return it as \"target_column\""
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
        "competition_metadata": comp_meta,
        "notebook_text": text_trunc
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    # 3) Second user message: output‐format instructions
    user_instructions = (
        "Please respond with a JSON object exactly with these keys (no extra keys!):\n"
        "{\n"
        '  "slug": "<competition_slug>",\n'
        '  "competition_problem_type": "classification|regression",\n'
        '  "competition_problem_subtype": (a single, concise phrase—lower-case words and hyphens only—describing the specific subtype, e.g. “binary classification”, “multiclass classification”, “multi-label classification”, “time-series forecasting”, “continuous regression”, “ordinal regression”, etc. or any other that fits.)\n"'
        '  "competition_problem_description": "Dense, short, and detailed description of the problem, what needs to be found, no repetitive words",\n'
        '  "competition_dataset_type": "choose exactly one primary data modality from this list (capitalized exactly): Tabular, Time-series, Text, Image, Audio, Video, Geospatial, Graph, Multimodal. ",\n'
        '  "competition_dataset_description": "based on the provided `dataset_schema` and `target schema` give a brief list of each feature name + its type",\n'
        '  "preprocessing_steps": [\n'
        '      "<step 1 in plain English>",\n'
        '      "<step 2>",\n'
        '      "…"\n'
        '  ],\n'
        '  "notebook_model_layers_code": "<complete code snippet defining each layer with all its parameters>",\n'
        '  "used_technique": "<\\"DL\\" or \\"ML\\">",\n'
        '  "library": "<library>",\n'
        '  "target_column": "<string>"\n'
        "}\n\n"
        "You also have access to `dataset_schema` and `target_summary` in the competition_metadata payload—use them to ground your explanations.\n"
        "**You have access to a variable** `dataset_columns` **which is the exact list of feature names.**\n"
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
    """
    1) For each Playground competition slug (147 total):
        a) Fetch competition metadata once (for LLM prompts).
        b) Run “kaggle kernels list ... -s tensorflow” and “-s pytorch” (CSV).
    2) For each candidate notebook (TF‐ or PT‐tagged), in descending vote order:
        a) Download the notebook locally if not already present.
        b) Read its content into text_content.
        c) Immediately call ask_llm_for_structured_output(comp_meta, notebook_path).
        d) If LLM says used_technique == "DL" AND library matches
            (“TensorFlow” for TF loop, “PyTorch” for PT loop), **then**:
            • write the ten‐field JSON output into structured_outputs.csv,
            • append to records, and increment tf_count or pt_count.
    3) Stop once you have max_per_keyword TF‐DL notebooks and max_per_keyword PT‐DL notebooks,
    or you exhaust all candidates.
    4) Return a DataFrame of all kept (DL) records, and also save them to Excel.

    The CSV structured_outputs.csv will contain one row per **kept** DL JSON, with columns:
    [
        competition_slug,
        competition_problem_type,
        competition_problem_subtype,
        competition_problem_description,
        competition_dataset_type,
        competition_dataset_description,
        notebook_description,
        used_technique,
        library,
        kernel_link
    ]
    """

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
            "competition_dataset_description",
            "preprocessing_steps",
            "notebook_model_layers_code",
            "used_technique",
            "library",
            "kernel_ref",
            "kernel_link"
        ])

    # Fetch slugs, either fresh or resuming
    if start:
        all_slugs = parse_playground_kaggle_from(start, num_competitions)
    else:
        all_slugs = parse_playground_kaggle(num_competitions)


    records = []
    driver = init_selenium_driver()

    for slug in all_slugs:
        print(f"\n[INFO] Processing competition: {slug}")
        comp_folder = Path("solutions") / slug
        comp_folder.mkdir(parents=True, exist_ok=True)

        # — 1) Scrape & parse HTML →
        html      = fetch_competition_page_html(slug, driver)
        comp_meta = parse_competition_metadata(html)
        comp_meta["slug"] = slug

        labels = label_competition(comp_meta)
        comp_meta.update(labels)
        print(comp_meta["target_column"])

        # — 3) Download & profile train.csv exactly once
        train_csv = comp_folder / "train.csv"
        if not train_csv.exists():
            kaggle_api.competition_download_file(
                slug, "train.csv", path=str(comp_folder)
            )

        profile = describe_schema(str(train_csv), comp_meta["target_column"])
        comp_meta["dataset_schema"]  = profile["dataset_schema"]
        comp_meta["target_summary"]  = profile["target_summary"]

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
            
                print("TARGET-TF: " + struct["target_column"])

                # Write that ten‐field JSON into our CSV (only DL entries)
                csv_writer.writerow([
                    struct["slug"],
                    struct["competition_problem_type"],
                    struct["competition_problem_subtype"],
                    struct["competition_problem_description"],
                    struct["competition_dataset_type"],
                    struct["competition_dataset_description"],
                    json.dumps(struct["preprocessing_steps"], ensure_ascii=False),
                    struct["notebook_model_layers_code"],
                    struct["used_technique"],
                    struct["library"],
                    kernel_ref, 
                    kernel_link
                ])

                # Build the combined record
                rec = {
                    "competition_slug":                struct["slug"],
                    "competition_problem_type":        struct["competition_problem_type"],
                    "competition_problem_subtype":     struct["competition_problem_subtype"],
                    "competition_problem_description": struct["competition_problem_description"],
                    "competition_dataset_type":        struct["competition_dataset_type"],
                    "competition_dataset_description": struct["competition_dataset_description"],
                    "preprocessing_steps":             struct["preprocessing_steps"],
                    "notebook_model_layers_code":      struct["notebook_model_layers_code"],
                    "used_technique":                  struct["used_technique"],   # "DL"
                    "library":                         struct["library"],          # "TensorFlow"
                    "kernel_ref":                      kernel_ref,
                    "kernel_link":                     kernel_link,
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
        
                print("TARGET-PYT: " + struct["target_column"])

                # Write that ten‐field JSON into our CSV
                csv_writer.writerow([
                    struct["slug"],
                    struct["competition_problem_type"],
                    struct["competition_problem_subtype"],
                    struct["competition_problem_description"],
                    struct["competition_dataset_type"],
                    struct["competition_dataset_description"],
                    json.dumps(struct["preprocessing_steps"], ensure_ascii=False),
                    struct["notebook_model_layers_code"],
                    struct["used_technique"],
                    struct["library"],
                    kernel_ref,
                    kernel_link
                ])

                rec = {
                    "competition_slug":                struct["slug"],
                    "competition_problem_type":        struct["competition_problem_type"],
                    "competition_problem_subtype":     struct["competition_problem_subtype"],
                    "competition_problem_description": struct["competition_problem_description"],
                    "competition_dataset_type":        struct["competition_dataset_type"],
                    "competition_dataset_description": struct["competition_dataset_description"],
                    "preprocessing_steps":             struct["preprocessing_steps"],
                    "notebook_model_layers_code":      struct["notebook_model_layers_code"],
                    "used_technique":                  struct["used_technique"],   # "DL"
                    "library":                         struct["library"],          # "PyTorch"
                    "kernel_ref":                      kernel_ref,
                    "kernel_link":                     kernel_link,
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

    return df_structured
