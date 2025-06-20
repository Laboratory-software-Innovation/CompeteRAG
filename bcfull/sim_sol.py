#By Illya Gordyy and ChatGPT


import os
import sys
import re
import json
import time
import pickle
import csv
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch

import faiss

from transformers import logging, AutoTokenizer, AutoModel
import transformers.modeling_utils as mutils
from transformers.utils import import_utils

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import InferenceApi
import openai
import tiktoken

from selenium_helper import init_selenium_driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from typing import Any, Dict, List
from typing import List, Tuple, Dict

# Monkey-patches
mutils.check_torch_load_is_safe = lambda *args, **kwargs: None
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
logging.set_verbosity_error()


# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration / Constants
# ─────────────────────────────────────────────────────────────────────────────

#  Where to store downloaded notebooks:
SOLUTIONS_DIR = Path("solutions")

# Training data for new competitions
COMP_TRAIN_DIR = Path("train")

#  Where to store TF-IDF indices, pickles, etc.:
INDEX_DIR = Path("index_data")

#  Where to store the structured output Excel:
EXCEL_FILE = Path("notebooks_and_competitions_structured.xlsx")

#  Kaggle API client
kaggle_api = KaggleApi()
kaggle_api.authenticate() 



#  OpenAI settings
load_dotenv()
OPENAI_MODEL = "gpt-4o-mini"
openai.api_key = os.getenv("OPENAI_API_KEY")
#  Tokenizer for truncation
ENCODER = tiktoken.get_encoding("cl100k_base")

#  Maximum tokens from a notebook to send to the LLM
MAX_NOTEBOOK_TOKENS = 1000


#  List of 147 Playground slugs (or however you get your list)
def parse_playground_kaggle(max_competitions: int = 9) -> list[str]:
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect Top‐Voted DL Notebooks (tensorflow & pytorch)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect Top-Voted DL Notebooks (tensorflow & pytorch), pruned of ML
# ─────────────────────────────────────────────────────────────────────────────

def download_csv(path: str, struct):
     # ───── Download train.csv ─────
    slug = struct["slug"]
    comp_folder = Path("solutions") / slug
    train_path = comp_folder / "train.csv"

    comp_folder.mkdir(parents=True, exist_ok=True)

    # download if missing
    if not train_path.exists():
        try:
            kaggle_api.competition_download_file(slug, "train.csv", path=str(comp_folder))
        except Exception as e:
            print(f"[WARN] Couldn't download train.csv for {slug}: {e}")
            return

    # profile (even if it was just downloaded)
    profile = describe_schema(str(train_path), struct["target_column"])
    if "dataset_schema" in profile:
        struct["dataset_schema"] = profile["dataset_schema"]
        struct["target_summary"] = profile["target_summary"]
    else:
        print(f"[WARN] Schema profiling failed for {slug}: {profile.get('error')}")


def collect_and_structured(max_per_keyword: int = 5) -> pd.DataFrame:
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

    # 1) Open CSV and write header (only DL entries will go in here)
    csv_file = open("notebooks_structured.csv", "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
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

    all_slugs = parse_playground_kaggle(2)
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scrape competition pages & ask LLM for structured output
# ─────────────────────────────────────────────────────────────────────────────

def truncate_to_token_limit(text: str, max_tokens: int = MAX_NOTEBOOK_TOKENS) -> str:
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODER.decode(tokens[:max_tokens])

def fetch_competition_page_html(slug: str, driver=None) -> str:
    """
    Return fully rendered HTML for a competition page.
    """
    if driver is None:
        driver = init_selenium_driver()
    url = f"https://www.kaggle.com/competitions/{slug}"
    driver.get(url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="competition-description"]'))
        )
    except:
        pass
    time.sleep(2)
    return driver.page_source

def parse_competition_metadata(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # 1) Title: first <h1>
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""


    # 3) Overview (Abstract) section
    desc_div = soup.find("div", id="abstract")
    overview = desc_div.get_text("\n").strip() if desc_div else ""

    # 4) Evaluation section
    eval_div = soup.find("div", id="evaluation")
    evaluation = eval_div.get_text("\n").strip() if eval_div else ""

    # 5) Dataset Description section
    dataset_desc = ""
    # find the <h2> that says "Dataset Description"
    dd_heading = soup.find("h2", string=re.compile(r"Dataset Description", re.IGNORECASE))
    if dd_heading:
        # container is a few levels up—grab the next sibling after its parent block
        container = dd_heading.find_parent("div", class_="sc-hRTuAS")  # or just find_parent("div")
        if container:
            # everything inside that container
            dataset_desc = container.get_text("\n").strip()
        else:
            # fallback: grab everything until the next <h2> or section break
            texts = []
            for sib in dd_heading.next_siblings:
                if sib.name and sib.name.startswith("h"):
                    break
                if getattr(sib, "get_text", None):
                    texts.append(sib.get_text("\n"))
            dataset_desc = "\n".join(texts).strip()

    # Build the Markdown‐style description
    parts = []
    if overview:
        parts.append("## Description\n\n" + overview)
    if evaluation:
        parts.append("## Evaluation\n\n" + evaluation)
    if dataset_desc:
        parts.append("## Dataset Description\n\n" + dataset_desc)

    full_desc = "\n\n".join(parts)

    return {
        "title": title,
        "problem_description": full_desc
    }

def describe_schema(
    url: str,
    class_col: str
) -> Dict[str, Any]:
    """
    Load a (possibly zipped) CSV/TSV from `url`, auto-sniff delimiter,
    use the python engine + skip malformed lines, then build your schema.
    """
    # —————————————————————
    # 0) If this is actually a ZIP archive, unzip it
    # —————————————————————
    # read magic header
    with open(url, "rb") as f:
        magic = f.read(4)
    if magic == b'PK\x03\x04':
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(url, 'r') as z:
            # find the first CSV inside
            for name in z.namelist():
                if name.lower().endswith('.csv'):
                    z.extract(name, tmpdir)
                    url = os.path.join(tmpdir, name)
                    break
        # now `url` points at the extracted CSV file

    # —————————————————————
    # 1) Sniff delimiter
    # —————————————————————
    try:
        with open(url, 'rb') as f:
            sample = f.read(2048)
        try:
            text = sample.decode('utf-8')
        except UnicodeDecodeError:
            text = sample.decode('latin1', errors='ignore')
        dialect = csv.Sniffer().sniff(text, delimiters=[',','\t',';'])
        sep = dialect.delimiter
    except Exception:
        sep = ','

    # —————————————————————
    # 2) Load with python engine, no quoting, skip broken lines
    # —————————————————————
    df = None
    last_err = None
    for enc in ("utf-8","utf-8-sig","latin1","ISO-8859-1"):
        try:
            df = pd.read_csv(
                url,
                sep=sep,
                encoding=enc,
                engine='python',
                quoting=csv.QUOTE_NONE,
                on_bad_lines='skip'
            )
            break
        except Exception as e:
            last_err = e
    if df is None:
        return {"error": f"Failed to load file (sep={sep!r}; tried encodings): {last_err}"}

    # —————————————————————
    # 3) Build the schema
    # —————————————————————
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()

    types: Dict[str,str] = {}
    for c in cols:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            types[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dt):
            types[c] = "datetime"
        else:
            nunique = df[c].nunique(dropna=True)
            types[c] = "categorical" if nunique < n_rows * 0.05 else "text"

    schema: List[Dict[str, Any]] = []
    for c in cols:
        entry = {"name": c, "type": types[c]}
        ser = df[c].dropna()
        if types[c] == "numeric":
            entry.update({
                "min": float(ser.min()),
                "median": float(ser.median()),
                "max": float(ser.max())
            })
        elif types[c] == "categorical":
            vc = ser.value_counts()
            entry.update({
                "cardinality": int(vc.size),
                "top": vc.index[:3].astype(str).tolist()
            })
        schema.append(entry)

    target_summary: Dict[str, Any] = {}
    if class_col in df.columns:
        ser = df[class_col].dropna()
        if types[class_col] == "numeric":
            stats = ser.describe()
            target_summary = {
                "min": float(stats["min"]),
                "median": float(stats["50%"]),
                "max": float(stats["max"])
            }
        else:
            pct = (ser.value_counts(normalize=True) * 100).round(2)
            target_summary = {str(k): float(v) for k,v in pct.items()}

    return {
        "source": url,
        "shape": {"rows": n_rows, "cols": n_cols},
        "dataset_schema": schema,
        "target_summary": target_summary
    }

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
        '  "competition_problem_description": based on the provided `dataset_schema` and `target schema` give a brief list of each feature name + its type'
        '  "competition_dataset_type": "<competition_dataset_type>",\n'
        '  "competition_dataset_description": "<competition_dataset_description>",\n'
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
# 3. Build TF-IDF indices (notebook descriptions & competition descriptions)
# ─────────────────────────────────────────────────────────────────────────────

def build_index(df_structured: pd.DataFrame,
                        model_name: str = "voidism/diffcse-roberta-base-sts"):
    """
    1) For each row, concatenate competition_problem_description,
    competition_dataset_description, and notebook_description into one string.
    2) Encode that combined text with a SentenceTransformer (DiffCSSE).
    3) One-hot encode the categorical columns:
    - competition_type
    - competition_problem_subtype
    - competition_dataset_type  (optional if always the same)
    - framework
    4) Concatenate the text embedding (e.g. 768-dim) with the one-hot vector.
    5) Save:
    - the SentenceTransformer model name (so we can reload it)
    - the OneHotEncoder object
    - the final N×D array of concatenated embeddings
    - the list of row IDs (e.g. kernel_ref) in the same order
    """

    INDEX_DIR.mkdir(exist_ok=True)

    # --- 1) Combine text fields per row ---
    # Assumes df_structured has these columns:
    #   competition_slug (or competition_id), competition_problem_description,
    #   competition_dataset_description, notebook_description,
    #   competition_type, competition_problem_subtype,
    #   competition_dataset_type, framework, kernel_ref (or notebook ID)

    # If some columns might be missing, verify or fillna("") before concatenating
    df = df_structured.copy()
    for col in ["competition_problem_description",
                "competition_dataset_description",
                #"notebook_description" - omit this for now, 
                # since a new given competition doesn't have a solution to compare to
                ]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    def _combine_text(row):
        parts = [
            row["competition_problem_description"].strip(),
            row["competition_dataset_description"].strip(),
            #row["notebook_description"].strip(), also since we are using preprocessing NLP and code segments
        ]
        # join with a separator; extra spaces don't harm SimCSE
        return " ".join(p for p in parts if p)

    df["combined_text"] = df.apply(_combine_text, axis=1)

    # --- 2) Encode combined_text with SentenceTransformer ---
    s_model = SentenceTransformer(model_name)  # uses the saved model_name DiffCSE
    texts = df["combined_text"].tolist()
    text_embeddings = s_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # --- 3) One‐Hot Encode categorical columns ---
    # Choose the categorical fields we want to include:
    cats = ["competition_type",
            "competition_problem_subtype",
            "competition_dataset_type",
            #"framework" since we don't have a solution we can also omit this
            ]
    # Fill missing categories with a placeholder (e.g., "Unknown")
    for c in cats:
        if c not in df.columns:
            df[c] = "Unknown"
        else:
            df[c] = df[c].fillna("Unknown")

    # Fit a single OneHotEncoder on all four columns at once
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
    cat_matrix = ohe.fit_transform(df[cats].values)
    # cat_matrix.shape == (N, C_total), e.g., (10000, 10)

    # --- 4) Concatenate text embeddings + one-hot vectors ---
    # If text_embeddings is float64, cast to float32 to save space
    if text_embeddings.dtype != np.float32:
        text_embeddings = text_embeddings.astype(np.float32)

    # Concatenate along the last axis: [ text_embed | cat_onehot ]
    combined_vectors = np.hstack([text_embeddings, cat_matrix])
    # combined_vectors.shape == (N, H + C_total)

    # --- 5) Save everything needed for later queries ---
    # 5.1) Save the SentenceTransformer model name (not the full weights)
    with open(INDEX_DIR / "text_encoder_model_name.txt", "w") as f:
        f.write(model_name)

    # 5.2) Save the OneHotEncoder object
    with open(INDEX_DIR / "onehot_encoder.pkl", "wb") as f:
        pickle.dump(ohe, f)

    with open(INDEX_DIR / "onehot_categories.pkl", "wb") as f:
        pickle.dump(ohe.categories_, f)

    # 5.3) Save the combined_vectors matrix (N×D)
    # Depending on N, D, we might want to use np.save or pickle
    np.save(INDEX_DIR / "combined_embeddings.npy", combined_vectors)
    

    # 5.4) Save the list of row IDs (e.g. kernel_ref) in the same order
    row_ids = df["kernel_ref"].tolist()
    with open(INDEX_DIR / "row_ids.pkl", "wb") as f:
        pickle.dump(row_ids, f)

    print(f"[INFO] Saved text‐encoder model name into {INDEX_DIR/'text_encoder_model_name.txt'}")
    print(f"[INFO] Saved OneHotEncoder (categorical) to {INDEX_DIR/'onehot_encoder.pkl'}")
    print(f"[INFO] Saved combined {combined_vectors.shape} embeddings to {INDEX_DIR/'combined_embeddings.npy'}")
    print(f"[INFO] Saved {len(row_ids)} row IDs to {INDEX_DIR/'row_ids.pkl'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute Similarities (each competition vs. all notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_from_description(
    desc_json: str,
    top_k: int = 5,
    exclude_competition: str = None
) -> None:
    """
    desc_json: path to a JSON file with exactly these keys:
      {
        "competition_problem_description": str,
        "competition_dataset_description": str,
        "competition_type": str,
        "competition_problem_subtype": str,
        "competition_dataset_type": str,
        "framework": str
      }
    """

    # ── 1) Load index artifacts ──────────────────────────────────────────
    model_name = (INDEX_DIR / "text_encoder_model_name.txt").read_text().strip()
    combined = np.load(INDEX_DIR / "combined_embeddings.npy")  # shape (N, D_total)
    with open(INDEX_DIR / "row_ids.pkl", "rb") as f:
        row_ids = pickle.load(f)                                # list of length N
    with open(INDEX_DIR / "onehot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)                                    # OneHotEncoder

    # compute dims
    C_total = sum(len(cats) for cats in ohe.categories_)
    text_dim = combined.shape[1] - C_total
    D_total = combined.shape[1]

    # ── 2) Reload the exact same SentenceTransformer ──────────────────
    s_model = SentenceTransformer(model_name)

    # ── 3) Read & combine new competition fields ────────────────────
    meta = json.loads(Path(desc_json).read_text(encoding="utf-8"))
    # 3a) text
    combined_text = " ".join([
        meta["competition_problem_description"].strip(),
        meta["competition_dataset_description"].strip()
    ])
    # 3b) cats
    cats_order = [
        "competition_type",
        "competition_problem_subtype",
        "competition_dataset_type",
        #"framework"
    ]
    cat_vals = [ meta[c] for c in cats_order ]

    # ── 4) Encode new query vector ──────────────────────────────────
    text_vec = s_model.encode(
        [combined_text],
        normalize_embeddings=True
    )[0]  # shape (text_dim,)
    cat_vec = ohe.transform([cat_vals])[0]  # shape (C_total,)
    query_vec = np.concatenate([text_vec, cat_vec], axis=0).astype("float32")  # (D_total,)

    # ── 5) Build & query FAISS HNSW index with cosine (inner‐product) ───
    # Make sure the vectors are unit‐length so inner‐product = cosine
    faiss.normalize_L2(combined)                              # (N, D_total)
    faiss.normalize_L2(query_vec.reshape(1, -1))              # (1, D_total)

    # Use METRIC_INNER_PRODUCT instead of the default METRIC_L2
    index = faiss.IndexHNSWFlat(D_total, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch       = 64

    index.add(combined)                                        # add all N normalized vectors
    buffer = 10 
    # Now D will be the cosine‐similarity in [-1,1], not an L2 distance
    D, I = index.search(query_vec.reshape(1, -1), top_k + buffer + (1 if exclude_competition else 0))
    indices = I[0]
    scores  = D[0]


    # ── 6) Optionally filter out same competition ───────────────────
    results = []
    if exclude_competition:
        # load structured CSV to map row_id→competition_slug
        df_nb = pd.read_csv("notebooks_structured.csv", usecols=["kernel_ref","competition_slug"])
        slug_map = dict(zip(df_nb["kernel_ref"], df_nb["competition_slug"]))
        for idx, score in zip(indices, scores):
            rid = row_ids[idx]
            if len(results) >= top_k:
                break
            if slug_map.get(rid) == exclude_competition:
                continue
            results.append((rid, float(score)))
            
    else:
        results = [(row_ids[idx], float(score)) for idx, score in zip(indices, scores)]

    # ── 7) Print top_k ───────────────────────────────────────────────
    print(f"\nTop {top_k} matches (excluding='{exclude_competition}'):\n")
    for rank, (rid, sc) in enumerate(results[:top_k], start=1):
        print(f"{rank:2d}. {rid}   (score={sc:.4f})")
    print()



def find_similar_ids(
    desc_json: str,
    top_k: int = 5,
    exclude_competition: str = None
) -> List[Tuple[str,float]]:
    """
    Same as find_similar_from_description but *returns* a list of
    (kernel_ref, score) instead of printing.
    """
    # ── load index artifacts ────────────────────────────
    combined = np.load(INDEX_DIR / "combined_embeddings.npy")
    with open(INDEX_DIR / "row_ids.pkl","rb") as f: row_ids = pickle.load(f)
    with open(INDEX_DIR / "onehot_encoder.pkl","rb") as f: ohe = pickle.load(f)

    # ── reload text‐encoder ──────────────────────────────
    model_name = (INDEX_DIR/"text_encoder_model_name.txt").read_text().strip()
    s_model = SentenceTransformer(model_name)

    # ── build query_vec just like before ────────────────
    meta = json.loads(Path(desc_json).read_text())

    # Safely extract descriptions, stringify non-strings
    prob_desc = meta.get("competition_problem_description", "")
    data_desc = meta.get("competition_dataset_description", "")
    if not isinstance(prob_desc, str):
        prob_desc = json.dumps(prob_desc, ensure_ascii=False)
    if not isinstance(data_desc, str):
        data_desc = json.dumps(data_desc, ensure_ascii=False)

    combined_text = prob_desc.strip() + "  " + data_desc.strip()
    text_vec = s_model.encode([combined_text], normalize_embeddings=True)[0]
    # combined_text = meta["competition_problem_description"].strip() + "  " + meta["competition_dataset_description"].strip()
    # text_vec = s_model.encode([combined_text], normalize_embeddings=True)[0]
    cat_vals = [meta[c] for c in ["competition_type","competition_problem_subtype","competition_dataset_type"]]
    cat_vec  = ohe.transform([cat_vals])[0]
    qv = np.concatenate([text_vec, cat_vec], axis=0).astype("float32")

    # ── normalize & search ──────────────────────────────
    faiss.normalize_L2(combined)
    faiss.normalize_L2(qv.reshape(1,-1))
    index = faiss.IndexHNSWFlat(combined.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(combined)

    # top-k
    D, I = index.search(qv.reshape(1,-1), top_k + (1 if exclude_competition else 0))
    results = []
    for idx, score in zip(I[0], D[0]):
        rid = row_ids[idx]
        if exclude_competition and rid.startswith(exclude_competition):
            continue
        results.append((rid, float(score)))
        if len(results) >= top_k:
            break

        
    return results


def solve_competition_with_examples(
    desc_json: str,
    train_csv_url: str,
    class_col: str,
    structured_csv: str = "notebooks_structured.csv",
    top_k: int = 5
) -> Dict[str,Any]:
    """
    1. Find the top_k similar notebooks.
    2. Download & profile train.csv → dataset_schema.
    3. Build & send a final LLM prompt with:
       - New comp metadata + dataset_schema
       - Summaries of those K notebooks
       - Instruction to propose a JSON solution outline
    """
    # 1) get top-k notebook refs + scores
    topk = find_similar_ids(desc_json, top_k=top_k, exclude_competition=None)

    # 2) load your structured outputs to fetch summaries
    df = pd.read_csv(structured_csv)
    example_texts = []
    for rank, (rid, score) in enumerate(topk, start=1):
        row = df[df["kernel_ref"] == rid].iloc[0]
        example_texts.append(
            f"{rank}. `{rid}` (score={score:.3f}): preprocessing={row['preprocessing_steps']}; "
            f"layers:\n{row['notebook_model_layers_code']}"
        )

    # 3) profile the new dataset
    schema_profile = describe_schema(train_csv_url, class_col)

    # 4) load new competition metadata
    new_meta = json.loads(Path(desc_json).read_text())

    # 5) build LLM messages
    system = {
        "role": "system",
        "content": (
            "You are an expert data scientist. "
            "Below is a new Kaggle competition and several example solutions. "
            "Using only these examples and the dataset schema, propose a complete modeling plan."
        )
    }
    user_new = {
        "role": "user",
        "content": (
            "New competition metadata:\n" +
            json.dumps(new_meta, ensure_ascii=False, indent=2) +
            "\n\nDataset schema:\n" +
            json.dumps(schema_profile, ensure_ascii=False, indent=2)
        )
    }
    user_examples = {
        "role": "user",
        "content": "Here are the top-K example notebooks:\n" + "\n".join(example_texts)
    }
    user_task = {
        "role": "user",
        "content": (
            "Based on the above examples and schema, outline in JSON:\n"
            "{\n"
            '  "architecture": "<model & rationale>",\n'
            '  "preprocessing": "<list of steps>",\n'
            '  "hyperparameters": "<learning rate, batch size, epochs, etc.>",\n'
            '  "extras": "<data aug, loss, training tricks>"\n'
            "}\n"
            "**Return only valid JSON.**"
        )
    }

    messages = [system, user_new, user_examples, user_task]

    # 6) call the LLM
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=messages
    )
    content = resp.choices[0].message.content.strip()

    # strip fences if any
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1]).strip()

    return json.loads(content)


def structure_and_label_competition(
    comp_meta: dict,
    dataset_schema: dict
) -> dict:
    """
    Returns exactly these keys:
      {
        "slug": …,
        "competition_type": …,
        "competition_problem_subtype": …,
        "competition_problem_description": …,
        "competition_dataset_type": …,
        "competition_dataset_description": …,
        "target_column": …
      }
    """
    system = {
      "role": "system",
      "content": (
        "You are an expert data scientist.  "
        "Below are the raw Kaggle competition metadata and a structured dataset schema.  "
        "Emit **only** a JSON object with exactly these keys:\n"
        "  slug                              (the competition slug)\n"
        "  competition_type          (e.g. “regression” or “classification”)\n"
        "  competition_problem_subtype       (e.g. “binary classification”)\n"
        "  competition_problem_description   (2–3 sentence summary)\n"
        "  competition_dataset_type          (e.g. “tabular”)\n"
        "  competition_dataset_description   (brief list of each feature name + its type)\n"
        "  target_column                     (exact name of the label column in train.csv)\n"
        "No extra fields, no markdown fences, just valid JSON."
      )
    }
    user = {
      "role": "user",
      "content": json.dumps({
        "competition_metadata": comp_meta,
        "dataset_schema":       dataset_schema
      }, ensure_ascii=False)
    }

    resp = openai.chat.completions.create(
      model=OPENAI_MODEL,
      temperature=0.0,
      messages=[system, user]
    )
    out = resp.choices[0].message.content.strip()
    if out.startswith("```"):
        out = "\n".join(out.split("\n")[1:-1]).strip()
    return json.loads(out)


def normalize_kernel_ref(ref: str) -> str:
    """
    Turn either
      - "username/kernel-name"
      - "https://www.kaggle.com/username/kernel-name"
    into exactly "username/kernel-name".
    """
    if ref.startswith("http"):
        # strip protocol+domain, drop any query-string
        ref = ref.split("://", 1)[-1]               # "www.kaggle.com/username/..."
        ref = ref.split("www.kaggle.com/", 1)[-1]   # "username/..."
        ref = ref.split("?", 1)[0]                 # drop any "?..."
    return ref

def solve_competition_with_code(
    class_col:       str,
    structured_csv:  str = "notebooks_structured.csv",
    top_k:           int = 5,
    kt:              bool = 1
) -> str:

    driver = init_selenium_driver()
    html   = fetch_competition_page_html(slug, driver)
    comp_meta = parse_competition_metadata(html)
    comp_meta["slug"] = slug
    driver.quit()


    # ── 2) Download & profile train.csv once
    comp_folder = Path("train") / slug
    comp_folder.mkdir(parents=True, exist_ok=True)
    train_csv = comp_folder / "train.csv"
    if not train_csv.exists():
        kaggle_api.competition_download_file(slug, "train.csv", path=str(comp_folder))

    profile = describe_schema(str(train_csv), class_col)
    
    print(profile)

    comp_struct = structure_and_label_competition(comp_meta, profile)
    desc_path = Path(f"{slug}_desc.json")
    desc_path.write_text(json.dumps(comp_struct, ensure_ascii=False, indent=2), encoding="utf-8")  



    # load & normalize your structured CSV
    df = pd.read_csv(structured_csv)
    df["kernel_ref_norm"] = df["kernel_ref"].apply(normalize_kernel_ref)

    # find top-K
    topk = find_similar_ids(str(Path(f"{slug}_desc.json")), top_k=top_k)    
    examples = []
    for rank, (kernel_ref, score) in enumerate(topk, start=1):
        kr = normalize_kernel_ref(kernel_ref)
        sub = df[df["kernel_ref_norm"] == kr]
        if sub.empty:
            print(f"[WARN] No entry for {kr!r}, skipping example {rank}")
            continue
        row = sub.iloc[0]
        prep_steps = row["preprocessing_steps"]
        layer_code = row["notebook_model_layers_code"]
        examples.append((rank, kr, score, prep_steps, layer_code))

    base_prompt = ("You are a world-class deep learning engineer and data scientist.  "
            "Generate **only runnable Python code** wrapped in <Code>…</Code> that builds a robust solution for any tabular competition with these requirements:\n\n"
            "1. Reproducibility:\n"
            "   - Set global seeds for Python, NumPy, and the chosen DL framework (TensorFlow or PyTorch), plus scikit-learn.\n"
            "2. Imports:\n"
            "   - pandas, numpy,\n"
            "   - sklearn.model_selection (train_test_split),\n"
            "   - sklearn.impute (SimpleImputer),\n"
            "   - sklearn.compose (ColumnTransformer),\n"
            "   - sklearn.preprocessing (StandardScaler, OneHotEncoder),\n"
            "   - sklearn.pipeline (Pipeline),\n"
            "   - tensorflow/torch,\n"
            "   - tensorflow.keras.callbacks (EarlyStopping, ModelCheckpoint) or torch equivalents,\n"
            "   - json, time.\n"
            "3. Data Loading:\n"
            "   - Read train.csv and test.csv.\n"
            "   - Identify and preserve the ID column (first column if name unknown).\n"
            "4. Feature Engineering:\n"
            "   - Drop or transform irrelevant columns as needed (e.g. extract Title from Name, Deck from Cabin, then drop raw fields).\n"
            "5. Train/Validation Split:\n"
            "   - Split into train/validation (80/20) with stratification on the target column.\n"
            "6. Preprocessing Pipeline:\n"
            "   - Auto-detect numeric vs. categorical features via df.select_dtypes.\n"
            "   - Build a ColumnTransformer that uses **sklearn.pipeline.Pipeline** for each group:\n"
            "       • Numeric pipeline: SimpleImputer(strategy='median', add_indicator=True) → StandardScaler().\n"
            "       • Categorical pipeline: SimpleImputer(strategy='most_frequent') → OneHotEncoder(sparse_output=False, handle_unknown='ignore').\n"
            "   - Fit_transform train and transform validation/test without reassigning back to DataFrame columns.\n"
            "7. Determine feature dimension:\n"
            "   - After fit_transform, set n_features = X_train_proc.shape[1] and use for model input_shape.\n"
            "8. Model Definition:\n"
            "   - Choose framework (Keras/TensorFlow **or** PyTorch) based on examples.\n"
            "   - Construct an ANN with at least two hidden layers (e.g. 64→32), including Dropout or BatchNorm.\n"
            "9. Compilation:\n"
            "   - Adam optimizer, binary_crossentropy (or appropriate) loss, and accuracy metric.\n"
            "10. Callbacks & Training:\n"
            "    - EarlyStopping(monitor='val_loss', patience=5) and ModelCheckpoint(save_best_only=True).\n"
            "    - Record start/end times to compute training duration (HH:MM:SS.ms).\n"
            "    - Train up to 100 epochs (or until early stop), verbose=1, with validation_data.\n"
            "11. Evaluation & Logging:\n"
            "    - Load best model weights, then extract final training_accuracy, training_loss, validation_accuracy, validation_loss.\n"
            "    - Save these plus elapsed time into results.json.\n"
            "12. Prediction & Submission:\n"
            "    - Transform test set via the same pipeline.\n"
            "    - Predict, threshold at 0.5 for binary targets.\n"
            "    - Write submission_result.csv using the preserved ID column and predicted target.\n\n"
            "Make no assumptions about specific column names beyond train.csv and test.csv and a single target column; auto-detect where needed.  "
            "Return only valid Python code between <Code> and </Code>.")
        # 2) Define the extra Keras-Tuner snippet
    tuner_prompt = (
        "### Hyperparameter Tuning (kt==1)\n"
        "10. **Search Space**\n"
        "    – hp.Int('num_layers',1,3)\n"
        "    – hp.Int('units_0',32,256,step=32)\n"
        "    – hp.Int('units_1',16,128,step=16)\n"
        "    – hp.Float('dropout',0.0,0.5,step=0.1)\n"
        "    – hp.Choice('learning_rate',[1e-2,1e-3,1e-4])\n"
        "    – hp.Int('batch_size',16,64,step=16)\n"
        "11. **HyperModel**\n"
        "```python\n"
        "class TabularHyperModel(HyperModel):\n"
        "    def build(self, hp):\n"
        "        num_layers = hp.Int('num_layers', 1, 3)\n"
        "        lr         = hp.Choice('learning_rate',[1e-2,1e-3,1e-4])\n"
        "        dropout    = hp.Float('dropout', 0.0, 0.5, step=0.1)\n"
        "        model = tf.keras.Sequential()\n"
        "        for i in range(num_layers):\n"
        "            units = hp.Int(f'units_{i}', 16, 128, step=16)\n"
        "            model.add(tf.keras.layers.Dense(units, activation='relu'))\n"
        "            model.add(tf.keras.layers.Dropout(dropout))\n"
        "        model.add(tf.keras.layers.Dense(1, activation='sigmoid' if classification else 'linear'))\n"
        "        model.compile(\n"
        "            optimizer=tf.keras.optimizers.Adam(lr),\n"
        "            loss='binary_crossentropy' if classification else 'mean_squared_error',\n"
        "            metrics=['accuracy'] if classification else ['RootMeanSquaredError']\n"
        "        )\n"
        "        return model\n\n"
        "    def fit(self, hp, model, x, y, **kwargs):\n"
        "        bs = hp.Int('batch_size', 16, 64, step=16)\n"
        "        return model.fit(x, y,\n"
        "                         batch_size=bs,\n"
        "                         callbacks=[EarlyStopping(monitor='val_loss', patience=5)],\n"
        "                         **kwargs)\n"
        "```\n"
        "12. **Tuner Setup**\n"
        "```python\n"
        "tuner = kt.Hyperband(\n"
        "    TabularHyperModel(),\n"
        "    objective='val_accuracy' if classification else 'val_root_mean_squared_error',\n"
        "    max_epochs=50,\n"
        "    factor=3,\n"
        "    directory='tuner_logs',\n"
        "    project_name='tabular_hb'\n"
        ")\n"
        "```\n"
        "13. **Search & Retrieve**\n"
        "```python\n"
        "tuner.search(\n"
        "    X_train_proc, y_train,\n"
        "    validation_data=(X_val_proc, y_val),\n"
        "    epochs=50\n"
        ")\n"
        "best_hps   = tuner.get_best_hyperparameters(1)[0]\n"
        "best_model = tuner.hypermodel.build(best_hps)\n"
        "```\n"
        "14. **Re-train Best Model**\n"
        "```python\n"
        "best_model.fit(\n"
        "    X_train_proc, y_train,\n"
        "    epochs=100,\n"
        "    batch_size=best_hps.get('batch_size'),\n"
        "    callbacks=[ModelCheckpoint('best.h5', save_best_only=True)],\n"
        "    validation_data=(X_val_proc, y_val)\n"
        ")\n"
        "```\n"
        "{% endif %}\n"
        "15. **Evaluation & Logging**\n"
        "    – Load best weights → eval train/val → save to results.json\n"
        "16. **Prediction & Submission**\n"
        "    – Transform test → predict → threshold (if classification) → write submission_result.csv\n"
    )


    system_content = base_prompt
    if kt:  # only append tuner text when kt == 1 / True
        print("KERAS-TUNER MODEL GENERATION")
        system_content += "\n\n" + tuner_prompt

        
    system = { "role": "system", "content": system_content }


    # new comp + schema
    user_parts = [
      "### New competition ###",
      json.dumps(comp_struct, indent=2, ensure_ascii=False),
      "\n### Dataset schema ###",
      json.dumps(profile, indent=2, ensure_ascii=False),
      "\n### Example summaries ###"
    ]


    # loop through the examples list you actually populated
    for rank, kr, sc, prep, layers in examples:
        user_parts.append(
            f"{rank}. `{kr}` (score={sc:.3f}):\n"
            f"   • preprocessing steps: {prep}\n"
            f"   • model layers code:\n{layers}\n"
        )
    user_parts.append(
      "\nNow write the full solution notebook in Python.  "
      "Return only the code (no markdown or comments)."
    )

    messages = [
      system,
      {"role":"user", "content":"\n".join(user_parts)}
    ]

    resp = openai.chat.completions.create(
      model=OPENAI_MODEL,
      temperature=0.0,
      messages=messages
    )
    code = resp.choices[0].message.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    return code


def followup_prompt(
    slug: str
) -> str:
    """
    Reads a solution file which contains marked sections:
      <Code> ... </Code>
      <Error> ... </Error>
    Sends these to the LLM in a follow-up prompt asking for corrected code.
    
    Returns the corrected code (without markers).
    """
    solution_path = str(str(slug) + "_solution.py")

    path = Path(solution_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {solution_path}")
    
    text = path.read_text(encoding="utf-8")

    # extract between markers
    code_match = re.search(r"<Code>(.*?)</Code>", text, re.S)
    err_match  = re.search(r"<Error>(.*?)</Error>", text, re.S)
    if not code_match or not err_match:
        raise ValueError("File must contain <Code>...</Code> and <Error>...</Error> sections")

    code_block = code_match.group(1).strip()
    error_msg  = err_match.group(1).strip()

    system = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer with an expertice in debugging the code.  "
            "Turn on the verbose and save the training and validtion accuracy and log of the last epoch in a json file (results.json). It will have the following keys: {training_accuracy, training_loss,validation_accuracy and validation_loss}  "
            "Now you will be given a deep learning <Code> along with the <Error> log. Think step by step and generate a fix for this code. Rewrite the full code from the begining, fixing the bug. In you code, include the code that records the time of how long the model trains. Write the code in this format"
            "<Code>"
            "Your code goes here"
            "</Code>"    
        )
    }
    user = {
        "role": "user",
        "content": (
            "<Code>\n"
            f"{code_block}\n"
            "</Code>\n\n"
            "<Error>\n"
            f"{error_msg}\n"
            "</Error>\n\n"
            "Return only the corrected Python code, wrapped in <Code>...</Code>."
        )
    }

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system, user]
    )
    reply = resp.choices[0].message.content.strip()
    # strip markers
    if reply.startswith("<Code>") and reply.endswith("</Code>"):
        return reply[len("<Code>"):-len("</Code>")].strip()
    return reply





# ─────────────────────────────────────────────────────────────────────────────
# 5. Command‐Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def print_usage():
    print("""
        Usage:
        python pipeline.py collect_and_structured
        python pipeline.py build_index
        python pipeline.py find_similar
        python pipeline.py find_similar_desc <description.txt> [notebooks|comps] [top_k]

        Examples:
        python pipeline.py find_similar_desc my_problem.txt notebooks 5
        python pipeline.py find_similar_desc my_problem.txt comps 10
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "collect_and_structured":
        df_struct = collect_and_structured(max_per_keyword=5)
        print(f"[OK] Collected and structured {len(df_struct)} notebooks.")
        sys.exit(0)

    elif cmd == "build_index":
        if not Path("notebooks_structured.csv").exists():
            print("[ERROR] Please run `collect_and_structured` first.")
            sys.exit(1)
        df_struct = pd.read_csv("notebooks_structured.csv")
        build_index(df_struct)
        sys.exit(0)
    elif cmd == "find_similar_desc":
        # Usage: python3 pipeline.py find_similar_desc <desc_and_meta.json> <top_k> [<exclude_competition>]
        if len(sys.argv) < 4:
            print("Usage: python3 pipeline.py find_similar_desc <desc_meta.json> <top_k> [<exclude_competition>]")
            sys.exit(1)

        desc_json    = sys.argv[2]
        top_k        = int(sys.argv[3])
        exclude_comp = sys.argv[4] if len(sys.argv) >= 5 else None

        find_similar_from_description(
            desc_json,
            top_k=top_k,
            exclude_competition=exclude_comp
        )
        sys.exit(0)

    elif cmd == "auto_solve_code":
        if len(sys.argv) < 4:
            print("Usage: python pipeline.py auto_solve_code <slug> <class_col> [top_k] [Keras-Tuner True:1|False:0]")
            sys.exit(1)
        slug      = sys.argv[2]
        class_col = sys.argv[3]
        top_k     = int(sys.argv[4]) if len(sys.argv)>4 else 5
        kt = int(sys.argv[5]) if len(sys.argv) > 5 else 0

        # call the new solver
        notebook_code = solve_competition_with_code(
            class_col     = class_col,
            structured_csv= "notebooks_structured.csv",
            top_k         = top_k,
            kt            = kt
        )
        # write out the notebook
        out_path = Path(f"{slug}_solution.py")
        out_path.write_text(notebook_code, encoding="utf-8")
        print(f"[OK] Solution code written to {out_path}")

    elif cmd == "followup":
        # Usage: python pipeline.py followup <solution_file.py>
        if len(sys.argv) != 3:
            print("Usage: python pipeline.py followup slug")
            sys.exit(1)
            
        slug = sys.argv[2]
        print(slug)
        try:
            corrected_code = followup_prompt(str(slug))
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        # Write out the corrected version alongside the original
        orig = Path(str(slug))
        fixed = orig.with_name(orig.stem + "_fixed.py")
        fixed.write_text(corrected_code, encoding="utf-8")
        print(f"[OK] Corrected code written to {fixed}")

    else:
        print_usage()
        sys.exit(1)

