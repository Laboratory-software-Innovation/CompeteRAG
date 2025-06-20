
import os
import re
import time
import tempfile
import zipfile
import csv
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from selenium_helper import init_selenium_driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from config import ENCODER, MAX_NOTEBOOK_TOKENS, kaggle_api

from typing import Any, Dict, List
from typing import List, Tuple, Dict


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




def truncate_to_token_limit(text: str, max_tokens: int = MAX_NOTEBOOK_TOKENS) -> str:
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODER.decode(tokens[:max_tokens])


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



"""
    Describe dataset 
"""
def describe_schema(
    url: str,
    class_col: str
) -> Dict[str, Any]:
    """
    Load a (possibly zipped) CSV/TSV from `url`, auto-sniff delimiter,
    use the python engine + skip malformed lines, then build your schema.
    """
    
    # 0) If this is actually a ZIP archive, unzip it
    
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

    
    # 1) Find delimiter
    
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

    
    # 2) Load with python engine, no quoting, skip broken lines
    
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

    
    # 3) Build the schema
    
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


