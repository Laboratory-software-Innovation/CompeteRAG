
import os
import re
import time
import tempfile
import zipfile
import csv
from pathlib import Path
import py7zr
import gzip

import pandas as pd
from bs4 import BeautifulSoup

from selenium_helper import init_selenium_driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from config import ENCODER, MAX_NOTEBOOK_TOKENS, kaggle_api

from typing import Any, List, Tuple, Dict, Union


# ─────────────────────────────────────────────────────────────────────────────
# Scrape competition pages & ask LLM for structured output
# ─────────────────────────────────────────────────────────────────────────────

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


def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def truncate_to_token_limit(text: str, max_tokens: int = MAX_NOTEBOOK_TOKENS) -> str:
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODER.decode(tokens[:max_tokens])

def parse_competition_data_tab(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    info = {}

    # 1) Dataset Description block
    dd_h2 = soup.find("h2", string=re.compile(r"Dataset Description", re.IGNORECASE))
    if dd_h2:
        # the <div class="sc-lhcVAQ fqIFbB"> immediately follows
        container = dd_h2.find_next("div", class_="sc-lhcVAQ")
        if container:
            info["dataset_description"] = container.get_text("\n", strip=True)

    # 2) summary stats: Files / Size / Type / License
    for label in ("Files", "Size", "Type", "License"):
        h2 = soup.find("h2", string=label)
        if h2:
            p = h2.find_next_sibling("p")
            info[label.lower()] = p.get_text(strip=True) if p else ""

    # 3) Data Explorer → list of filenames
    de_h2 = soup.find("h2", string=re.compile(r"Data Explorer", re.IGNORECASE))
    files = []
    if de_h2:
        ul = de_h2.find_next("ul")
        if ul:
            for li in ul.find_all("li"):
                # the filename lives in the <p> inside each <li>
                p = li.find("p")
                if p:
                    files.append(p.get_text(strip=True))
    info["files_list"] = files

    return info


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
        "competition_metadata": full_desc, 
    }



"""
    Describe dataset 
"""

def extract_tabular(file_path: str) -> str:
    """
    If file_path is an archive (.zip, .csv.zip, .tsv.zip, .7z, .gz),
    extract the first .csv or .tsv inside a temp dir and return its path.
    Otherwise return file_path unchanged.
    """
    # read magic bytes
    with open(file_path, 'rb') as f:
        magic = f.read(6)
    temp_dir = tempfile.mkdtemp()

    # ZIP archives (.zip, .csv.zip, .tsv.zip, .zip)
    if magic[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(file_path, 'r') as z:
            members = [n for n in z.namelist() if n.lower().endswith(('.csv', '.tsv'))]
            if not members:
                raise RuntimeError(f"No .csv or .tsv found in ZIP {file_path}")
            member = members[0]
            z.extract(member, temp_dir)
            return os.path.join(temp_dir, member)

    # 7z archives (.7z)
    if magic == b'7z\xBC\xAF\'\x1C':
        with py7zr.SevenZipFile(file_path, 'r') as z:
            members = [n for n in z.getnames() if n.lower().endswith(('.csv', '.tsv'))]
            if not members:
                raise RuntimeError(f"No .csv or .tsv found in 7z {file_path}")
            member = members[0]
            z.extract(targets=[member], path=temp_dir)
            return os.path.join(temp_dir, member)

    # gzip (.gz)
    if magic[:2] == b'\x1f\x8b':
        base = os.path.basename(file_path)[:-3]  # strip .gz
        out_path = os.path.join(temp_dir, base)
        with gzip.open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        return out_path

    # not an archive: assume plain .csv or .tsv
    return file_path





def describe_schema(
    source_path: str,
    target_column: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    1) Unpack archives or compressed files if needed
    2) Auto-detect delimiter (comma, tab, semicolon), fallback to tab for .tsv
    3) Load with pandas (python engine, skip bad lines)
    4) Build a schema and target summary for one or more columns
    """
    # 0) normalize target list
    if isinstance(target_column, str):
        targets = [target_column]
    else:
        targets = list(target_column)

    # 1) unpack if needed
    csv_path = extract_tabular(source_path)

    # 2) sniff delimiter
    with open(csv_path, 'rb') as f:
        sample = f.read(2048)
    try:
        text = sample.decode('utf-8')
    except UnicodeDecodeError:
        text = sample.decode('latin1', errors='ignore')
    try:
        sep = csv.Sniffer().sniff(text, delimiters=[',','\t',';']).delimiter
    except csv.Error:
        sep = '\t' if csv_path.lower().endswith('.tsv') else ','

    # 3) load DataFrame
    for enc in ("utf-8","utf-8-sig","latin1","ISO-8859-1"):
        try:
            df = pd.read_csv(
                csv_path, sep=sep, encoding=enc,
                engine='python', quoting=csv.QUOTE_NONE,
                on_bad_lines='skip'
            )
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read {csv_path}")

    # 4) basic info
    n_rows, n_cols = df.shape

    # 5) compute missing
    missing_counts = df.isnull().sum()

    # 6) build feature list
    features: List[Dict[str,Any]] = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            t = "int"
        elif pd.api.types.is_float_dtype(dtype):
            t = "float"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            t = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            t = "boolean"
        else:
            t = "string"

        miss_pct = round(float(missing_counts[col]) / n_rows * 100, 2)
        is_tgt = col in targets

        features.append({
            "name": col,
            "type": t,
            "missing_pct": miss_pct,
            "is_target": is_tgt
        })

    # 7) build target summary
    target_summary: Dict[str,Any] = {}
    for tgt in targets:
        if tgt not in df.columns:
            target_summary[tgt] = {"error": "not found"}
            continue

        ser = df[tgt].dropna()
        if pd.api.types.is_numeric_dtype(df[tgt].dtype):
            stats = ser.describe()
            target_summary[tgt] = {
                "min": float(stats["min"]),
                "median": float(stats["50%"]),
                "max": float(stats["max"])
            }
        else:
            vc = (ser.value_counts(normalize=True) * 100).round(2)
            # simple dict of class → percent
            target_summary[tgt] = {str(k): float(v) for k,v in vc.items()}

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "dataset_schema": features,
        "target_summary": target_summary
    }

def download_train_file(slug: str, path):
    comp_folder = Path(path) / slug
    comp_folder.mkdir(parents=True, exist_ok=True)

    check_exts = [
        ".csv", ".tsv",
        ".csv.zip", ".tsv.zip",
        ".csv.gz",  ".tsv.gz"
    ]

    # 1) see if any already exist
    for ext in check_exts:
        candidate = comp_folder / f"train{ext}"
        if candidate.exists():
            return candidate

    # 2) none exist → list competition files & download the first match
    files = kaggle_api.competition_list_files(slug).files
    lc_to_orig = {f.name.lower(): f.name for f in files}
    for ext in check_exts:
        key = f"train{ext}"
        if key in lc_to_orig:
            kaggle_api.competition_download_file(slug, lc_to_orig[key], path=str(comp_folder))
            return comp_folder / lc_to_orig[key]

    # 3) still nothing → warn
    print(f"[WARN] No train file found for {slug}")
    return None


def download_csv(path: str, struct: dict):
    slug = struct["slug"]
    comp_folder = Path("solutions") / slug
    comp_folder.mkdir(parents=True, exist_ok=True)

    raw_file = download_train_file(slug, comp_folder)

    # 4) profile it (describe_schema will unpack any .zip/.gz/.tsv as needed)
    profile = describe_schema(str(raw_file), struct["target_column"])
    if "dataset_schema" in profile:
        struct["dataset_schema"] = profile["dataset_schema"]
        struct["target_summary"]  = profile["target_summary"]
    else:
        print(f"[WARN] Schema profiling failed for {slug}: {profile.get('error')}")
