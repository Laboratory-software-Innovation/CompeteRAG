import os
import re
import time
import tempfile
import zipfile
import csv
from pathlib import Path
import py7zr
import gzip
import openai

import pandas as pd
from bs4 import BeautifulSoup

from selenium_helper import init_selenium_driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from config import kaggle_api

from typing import Any, List, Tuple, Dict, Union,Optional


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

    return {
        "dataset_metadata": info, 
        "files_list": files
    }


def parse_competition_metadata(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    def grab_section(heading_text):
        hdr = soup.find(
            lambda tag: tag.name in ("h1","h2","h3")
                        and tag.get_text(strip=True).lower().startswith(heading_text.lower())
        )
        if not hdr:
            return ""
        parts = []
        for sib in hdr.next_siblings:
            # stop at next heading
            if sib.name and re.match(r"h[1-3]", sib.name, re.I):
                break
            if hasattr(sib, "get_text"):
                text = sib.get_text(" ", strip=True)
                if text:
                    parts.append(text)
        text = " ".join(parts)
        # collapse multiple spaces/newlines
        return re.sub(r"\s{2,}", " ", text).strip()

    # 1) Title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # 2) Sections
    overview = grab_section("Overview") or grab_section("Description") or ""
    evaluation = grab_section("Evaluation")
    dataset_desc = grab_section("Dataset Description")
    submission_fmt = grab_section("Submission File")

    # 3) Build the combined markdown (if you still want it)
    md_parts = []
    if overview:
        md_parts.append("## Description\n\n" + overview)
    if evaluation:
        md_parts.append("## Evaluation\n\n" + evaluation)
    if dataset_desc:
        md_parts.append("## Dataset Description\n\n" + dataset_desc)
    if submission_fmt:
        md_parts.append("## Submission File\n\n" + submission_fmt)
    combined_md = "\n\n".join(md_parts)

    return {
        "title": title,
        "competition_metadata": combined_md
    }



"""
    Describe dataset 
"""

def extract_tabular(file_path: str) -> Tuple[Optional[tempfile.TemporaryDirectory], str]:
    """
    If file_path is an archive (.zip, .csv.zip, .tsv.zip, .7z, .gz),
    extract the first .csv or .tsv inside a temp dir and return its path.
    Otherwise return file_path unchanged.
    """
    # read magic bytes
    with open(file_path, 'rb') as f:
        magic = f.read(6)


    # ZIP archives (.zip, .csv.zip, .tsv.zip, .zip)
    if magic[:4] == b'PK\x03\x04' or file_path.lower().endswith(('.zip', '.csv.zip', '.tsv.zip')):
        z = zipfile.ZipFile(file_path, 'r')
        members = [n for n in z.namelist() if n.lower().endswith(('.csv', '.tsv'))]
        if not members:
            raise RuntimeError(f"No .csv or .tsv found in ZIP {file_path}")
        member = members[0]
        temp_dir = tempfile.TemporaryDirectory()
        z.extract(member, temp_dir.name)
        return temp_dir, os.path.join(temp_dir.name, member)
    
    # 7z archives (.7z)
    if magic == b'7z\xBC\xAF\'\x1C' or file_path.lower().endswith('.7z'):
        temp_dir = tempfile.TemporaryDirectory()
        with py7zr.SevenZipFile(file_path, 'r') as z:
            members = [n for n in z.getnames() if n.lower().endswith(('.csv', '.tsv'))]
            if not members:
                raise RuntimeError(f"No .csv or .tsv found in 7z {file_path}")
            member = members[0]
            z.extract(targets=[member], path=temp_dir.name)
        return temp_dir, os.path.join(temp_dir.name, member)
    
    # gzip (.gz)
    if magic[:2] == b'\x1f\x8b':
        temp_dir = tempfile.TemporaryDirectory()
        base = os.path.basename(file_path)[:-3]
        out_path = os.path.join(temp_dir.name, base)
        with gzip.open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        return temp_dir, out_path


    # not an archive: assume plain .csv or .tsv
    return None, file_path



def compact_profile_for_llm(
    profile: Dict[str,Any],
    max_features: int = 50,
    max_classes: int  = 50
) -> Dict[str,Any]:
    """
    Collapse long schema runs (as you already do) *and* collapse/
    truncate target_summary so that classes like "123-F_1_2" are
    grouped by feature (F_1) into ranges, with count.  Any leftover
    non-matching classes fall back to top-K + "... more".
    """
    # 1) collapse schema exactly as before
    schema = profile["dataset_schema"]
    if len(schema) <= max_features:
        collapsed_schema = schema
    else:
        # your run‐detection & collapse logic here...
        # (for brevity, imagine you’ve copied in your existing code)
        collapsed_schema = _collapse_schema_runs(schema)

    # 2) collapse target_summary
    ts = profile["target_summary"]
    truncated_ts: Dict[str,Any] = {}
    # regex to pull out "<row>-<feature>_<idx>"
    cls_pat = re.compile(r"^\d+-(F_\d+)_(\d+)$")

    for tgt_col, dist in ts.items():
        if isinstance(dist, dict) and all(isinstance(v, (int,float)) for v in dist.values()):
            # group by feature prefix
            groups: Dict[str, List[int]] = {}
            others: Dict[str, float] = {}
            for cls, pct in dist.items():
                m = cls_pat.match(cls)
                if m:
                    feat, idx = m.group(1), int(m.group(2))
                    groups.setdefault(feat, []).append(idx)
                else:
                    others[cls] = pct

            collapsed_classes: Dict[str, Any] = {}
            # first collapse all the feature‐based groups
            for feat, idxs in groups.items():
                lo, hi = min(idxs), max(idxs)
                n = len(idxs)
                collapsed_classes[f"{feat}_{lo}–{feat}_{hi} ({n} classes)"] = None

            # then handle any “others” with top‐K + "... more"
            if others:
                # sort descending
                items = sorted(others.items(), key=lambda x: x[1], reverse=True)
                for cls, pct in items[:max_classes]:
                    collapsed_classes[cls] = pct
                rem = len(items) - max_classes
                if rem > 0:
                    collapsed_classes[f"... +{rem} more"] = None

            truncated_ts[tgt_col] = collapsed_classes

        else:
            # numeric summary, leave as is
            truncated_ts[tgt_col] = dist

    return {
        "shape":          profile["shape"],
        "dataset_schema": collapsed_schema,
        "target_summary": truncated_ts
    }

#Helper for large schemas
def _collapse_schema_runs(schema: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    pat = re.compile(r"^(.*?)(\d+)$")
    runs, current = [], [schema[0]]
    for feat in schema[1:]:
        prev = current[-1]
        m1, m2 = pat.match(prev["name"]), pat.match(feat["name"])
        if (
            feat["type"] == prev["type"]
            and m1 and m2
            and m1.group(1) == m2.group(1)
            and int(m2.group(2)) == int(m1.group(2)) + 1
        ):
            current.append(feat)
        else:
            runs.append(current)
            current = [feat]
    runs.append(current)

    collapsed = []
    for run in runs:
        t = run[0]["type"]
        if len(run) == 1:
            f = run[0]
            collapsed.append({
                "name":     f["name"],
                "type":        t,
                "missing_pct": f.get("missing_pct"),
                "is_target":   f.get("is_target", False)
            })
        else:
            start, end = run[0]["name"], run[-1]["name"]
            missing_avg = round(sum(f.get("missing_pct",0) for f in run) / len(run), 2)
            collapsed.append({
                "name":     f"{start}–{end} ({len(run)} cols)",
                "type":        t,
                "missing_pct": missing_avg,
                "is_target":   any(f.get("is_target", False) for f in run)
            })
    return collapsed


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
    temp_ctx, csv_path = extract_tabular(source_path)

    # 2) sniff delimiter
    try:
        # sniff delimiter
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

        # load
        df = None
        for enc in ("utf-8","utf-8-sig","latin1","ISO-8859-1"):
            try:
                df = pd.read_csv(
                    csv_path, sep=sep, encoding=enc,
                    engine='python', quoting=csv.QUOTE_NONE,
                    on_bad_lines='skip'
                )
                break
            except Exception:
                continue
        if df is None:
            raise RuntimeError(f"Failed to read {csv_path}")

        # shape & missing
        n_rows, n_cols = df.shape
        missing = df.isnull().sum()

        # features
        features: List[Dict[str, Any]] = []
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
            miss_pct = round(float(missing[col]) / n_rows * 100, 2)
            features.append({
                "name": col,
                "type": t,
                "missing_pct": miss_pct,
                "is_target": col in targets
            })

        # target summary
        target_summary: Dict[str, Any] = {}
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
                target_summary[tgt] = {str(k): float(v) for k, v in vc.items()}

        return {
            "shape": {"rows": n_rows, "cols": n_cols},
            "dataset_schema": features,
            "target_summary": target_summary
        }

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

def download_train_file(slug: str, path, files_list: List[str]) -> List[Path]:
    comp_folder = Path(path) / slug
    comp_folder.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for fname in files_list:
        # skip if we already have it
        dest = comp_folder / fname
        if not dest.exists():
            kaggle_api.competition_download_file(
                slug,
                fname,
                path=str(comp_folder)
            )
        local_paths.append(dest)
    return local_paths


def select_hyperparameter_profile(comp_meta, hyperparameter_bank):
    """
    comp_meta is a dict containing:
      - "competition_problem_type"
      - "competition_problem_subtype"
      - "data_profiles": filename → {
            "dataset_schema": [ {name, type, …}, … ],
            …
        }
    """
    # Build metadata tags
    tags = {
        comp_meta["competition_problem_type"],
        comp_meta["competition_problem_subtype"],
    }

    # Take one profile to infer modality/feature count
    sample_profile = next(iter(comp_meta["data_profiles"].values()))

    # **Convert** the list-of-dicts into a dict: name → type
    columns = {
        entry["name"]: entry["type"]
        for entry in sample_profile.get("dataset_schema", [])
    }

    # 1) modality tag
    if any(
        t.startswith("object") or n.lower().endswith(("text", "comment"))
        for n, t in columns.items()
    ):
        tags.add("text")
    elif any(t in ("image_path", "filepath") for t in columns.values()):
        tags.add("image")
    elif any(t.startswith("datetime") for t in columns.values()):
        tags.add("time-series")
    else:
        tags.add("tabular")

    # 2) feature‐count tag (subtract targets)
    n_feats = len(columns) - len(comp_meta.get("target_column", []))
    if n_feats < 10:
        tags.add("low_features")
    elif n_feats < 500:
        tags.add("medium_features")
    else:
        tags.add("high_features")

    # 3) missing‐values tag
    # your schema uses "missing_pct" per column
    if any(
        entry.get("missing_pct", 0) > 0
        for entry in sample_profile.get("dataset_schema", [])
    ):
        tags.add("missing-values")

    # Score each profile
    best_key, best_score = None, -1
    for key, prof in hyperparameter_bank.items():
        score = len(tags & set(prof["tags"]))
        if score > best_score:
            best_key, best_score = key, score


    print(comp_meta["competition_problem_type"])
    print(comp_meta["competition_problem_subtype"]) 
    print(best_key) 

    return best_key
