import json
import re
import pandas as pd
from pathlib import Path
from utils import extract_tabular, download_train_file

from typing import List, Dict, Optional
import openai

from config import OPENAI_MODEL,kaggle_api
from selenium_helper import init_selenium_driver
from utils import fetch_competition_page_html, parse_competition_metadata, parse_competition_data_tab, describe_schema, compact_profile_for_llm
from similarity import find_similar_ids
from prompts import generate_solution_schema,structure_and_label_competition_schema
from config import kaggle_api

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


"""
    New competition structure 
"""

def structure_and_label_competition(comp_meta: dict) -> dict:
    # 1) system & user messages
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert data scientist.  "
            "Below are the raw Kaggle competition metadata, dataset metadata, and a list of files.  "
            "Emit **only** a JSON object with exactly the keys specified in the function schema."
        )
    }
    user_msg = {
        "role": "user",
        "content": json.dumps({
            "competition_metadata": comp_meta["competition_metadata"],
            "dataset_metadata":     comp_meta["dataset_metadata"],
            "files_list":           comp_meta["files_list"],
            "all_files":            comp_meta.get("all_files", comp_meta["files_list"])
        }, ensure_ascii=False)
    }

    # 2) call the model with our function schema
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system_msg, user_msg],
        functions=[structure_and_label_competition_schema],
        function_call={"name": "structure_and_label_competition_schema"}
    )

    # 3) parse the function call
    message = response.choices[0].message
    args = json.loads(message.function_call.arguments)

    return args



def list_files(slug : str) -> List[str]:  
    all_names: List[str] = []
    next_page_token: Optional[str] = None

    while True:
        resp = kaggle_api.competition_list_files(
            competition=slug,
            page_size=200,
            page_token=next_page_token
        )
        all_names += [f.name for f in resp.files]

        next_page_token = getattr(resp, 'next_page_token', None)
        if not next_page_token:
            break

    return all_names



"""
    Initial prompt for Keras and Keras-Tuner  
"""

def solve_competition_with_code(
    slug:            str, 
    structured_csv:  str = "notebooks_structured.csv",
    top_k:           int = 5,
    kt:              bool = 0, 
) -> str:

    driver = init_selenium_driver()
    html   = fetch_competition_page_html(slug, driver)
    comp_meta = parse_competition_metadata(html)
    comp_meta["slug"] = slug


    # now fetch & parse the /data tab
    data_html = fetch_competition_page_html(f"{slug}/data", driver)
    temp = parse_competition_data_tab(data_html)   
    comp_meta["dataset_metadata"] = temp["dataset_metadata"]
    comp_meta["files_list"] = temp["files_list"]
    driver.quit()

    comp_folder = Path("test") / slug
    comp_folder.mkdir(parents=True, exist_ok=True)
    all_files = list_files(slug)
    comp_meta["all_files"] = all_files
 

    comp_struct = structure_and_label_competition(comp_meta)
    print("----------------")
    print(f"{slug}")
    print(comp_struct["files_list"])
    print(comp_struct["training_files"])
    print("----------------")
    downloaded_paths = download_train_file(
        comp_meta["slug"],
        comp_folder,
        comp_struct["training_files"]
    )

    # 3) profile each one
    all_schemas = {}
    for p in downloaded_paths:
        prof = describe_schema(
        source_path=str(p),
        target_column=comp_struct["target_column"]
        )

        # 2) collapse both the schema *and* the target_summary if they’re too big
        compacted = compact_profile_for_llm(
            prof,
            max_features=50,    # collapse runs if > n features
            max_classes=50       # keep top n classes per target
        )

        # 3) stash under the filename
        all_schemas[p.name] = compacted


    # now we have a dict mapping each filename → its profile
    comp_struct["data_profiles"] = all_schemas


    
    desc_path = Path(f"{comp_folder}/{slug}_desc.json")
    desc_path.write_text(json.dumps(comp_struct, ensure_ascii=False, indent=2), encoding="utf-8")  

    # load & normalize the structured CSV
    df = pd.read_csv(structured_csv)
    df["kernel_ref_norm"] = df["kernel_ref"].apply(normalize_kernel_ref)

    # find top-K
    topk = find_similar_ids(str(Path(f"{comp_folder}/{slug}_desc.json")), top_k=top_k)    
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

    

    payload = {
        "competition_slug":                 slug,
        "competition_problem_description":  comp_struct["competition_problem_description"],
        "competition_problem_type":         comp_struct["competition_type"],
        "competition_problem_subtype":      comp_struct["competition_problem_subtype"],
        "dataset_metadata":                 comp_struct["dataset_metadata"],
        "data_profiles":                    all_schemas,
        "files_preprocessing_instructions": comp_struct["files_preprocessing_instructions"],
        "target_columns":                    comp_struct["target_column"], 
        "training_files":                   comp_struct["training_files"],
        "all_files":                        all_files,
        "examples":                         [
            {
              "rank":               rank,
              "kernel_ref":         kr,
              "score":              sc,
              "preprocessing_steps": prep,
              "model_layers_code":   layers
            }
            for (rank, kr, sc, prep, layers) in examples
        ],
        "use_kt":                           bool(kt)
    }

    system_msg = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer and data scientist.  "
            "When called, generate only runnable Python code wrapped in <Code>…</Code>."
            "Emit ONLY a single JSON object with exactly one field: notebook_code"
        )
    }

    user_msg = {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False)
    }

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system_msg, user_msg],
        functions=[generate_solution_schema],
        function_call={"name": "generate_solution_schema"}
    )

    msg    = response.choices[0].message
    result = json.loads(msg.function_call.arguments)
    code   = result["notebook_code"]
    if code.startswith("<Code>") and code.endswith("</Code>"):
        code = code[len("<Code>"):-len("</Code>")].strip()
    
    return code 



"""
    Follow-up prompt
"""

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


