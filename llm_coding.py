import json
import re
import pandas as pd
from pathlib import Path
from utils import extract_tabular, download_train_file

from typing import List, Dict, Optional
import openai

from collections import defaultdict
from config import OPENAI_MODEL,kaggle_api
from selenium_helper import init_selenium_driver
from utils import fetch_competition_page_html, parse_competition_metadata, parse_competition_data_tab, describe_schema, compact_profile_for_llm,select_hyperparameter_profile
from similarity import find_similar_ids
from prompts import tools, structure_and_label_competition_schema,generate_tuner_schema
from config import kaggle_api, MAX_CLASSES, MAX_FEATURES
from tuner_bank import HYPERPARAMETER_BANK




def normalize_kernel_ref(ref: str) -> str:
    """
    Turn either
      - "username/kernel-name"
      - "https://www.kaggle.com/username/kernel-name"
    into exactly "username/kernel-name".
    """
    if ref.startswith("http"):
        # strip protocol+domain, drop any query-string
        ref = ref.split("://", 1)[-1]              
        ref = ref.split("www.kaggle.com/", 1)[-1]   
        ref = ref.split("?", 1)[0]                 
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
            "files_list":           comp_meta["files_list"], #Retrieved via parsing
            "all_files":            comp_meta["all_files"] #Retrieved via Kaggle API
        }, ensure_ascii=False)
    }

    #Parsing the files seemed to provide more useful file, due to Kaggle API output limitations and LLM prompt limit

    # 2) call the model with our function schema
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system_msg, user_msg],
        functions=[structure_and_label_competition_schema],
        function_call={"name": "structure_and_label_competition_schema"}
    )

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



def postprocess_code(code: str) -> str:
    """
    Extract just the code between <Code>…</Code> and drop any surrounding text.
    """
    # Find the content between the <Code> tags (including multiline)
    m = re.search(r"<Code>(.*?)</Code>", code, flags=re.DOTALL)
    if not m:
        # no tags found, just return the original
        return code
    # Return the inner snippet, trimmed of leading/trailing whitespace
    return m.group(1).strip()

def compact(comp_struct):
        raw_targets = comp_struct["target_column"]

        suffix_pat = re.compile(r"^(.+?)(\d+)$")

        parsed = []
        literal_targets = []

        for t in raw_targets:
            m = suffix_pat.match(t)
            if m:
                prefix, idx_str = m.group(1), m.group(2)
                try:
                    idx = int(idx_str)
                except ValueError:
                    literal_targets.append(t)
                    continue
                parsed.append((prefix, idx))
            else:
                literal_targets.append(t)

        # group parsed by prefix
        groups = defaultdict(list)
        for prefix, idx in parsed:
            groups[prefix].append(idx)

        # build range specs
        range_specs = []
        for prefix, idxs in groups.items():
            lo, hi = min(idxs), max(idxs)
            range_specs.append({
                "prefix":    prefix,
                "min_index": lo,
                "max_index": hi,
                "count":     len(idxs)
            })

        return range_specs


def generate_keras_schema_impl(
    competition_problem_description: str,
    competition_problem_subtype: str,
    dataset_metadata: str,
    data_profiles: dict,
    files_preprocessing_instructions: str,
    submission_example: str,
    files_list: list[str],
    examples: list[dict]
) -> str:
    # STEP 1: ask the Responses API to plan a function call
    plan_resp = openai.responses.create(
        model="o4-mini",
        input=[
            {"role":"system", "content":
             "You are a world-class deep learning engineer.  "
             "Decide to call generate_keras_schema to build a Keras notebook."
            },
            {"role":"user", "content": json.dumps({
                "competition_problem_description":  competition_problem_description,
                "competition_problem_subtype":      competition_problem_subtype,
                "dataset_metadata":                 dataset_metadata,
                "data_profiles":                    data_profiles,
                "files_preprocessing_instructions": files_preprocessing_instructions,
                "submission_example":               submission_example,
                "files_list":                       files_list,
                "examples":                         examples,
            }, ensure_ascii=False)}
        ],
        tools=tools,
        store=True
    )

    # look for a tool_call, not a function_call
    call_evt = next(
        ev for ev in plan_resp.output
        if ev.type == "function_call" and ev.name == "generate_keras_schema"
    )
    tool_inputs = json.loads(call_evt.arguments)


    # STEP 3: actually generate your notebook code 
    chat = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role":"system", "content":"Generate only the notebook code wrapped in <Code>…</Code>."},
            {"role":"user",   "content": json.dumps(tool_inputs, ensure_ascii=False)}
        ]
    )
    notebook_code = chat.choices[0].message.content

    # STEP 4: feed back only the function_call_output to finish the response
    wrapup = openai.responses.create(
        model="o4-mini",
        previous_response_id=plan_resp.id,
        input=[{
            "type":    "function_call_output",
            "call_id": call_evt.call_id,      
            "output":  notebook_code
        }],
        store=True,
        tools=tools
    )

    return wrapup.output_text

    
"""
    Initial prompt for Keras  
"""

def solve_competition_keras(
    slug:            str, 
    structured_csv:  str = "notebooks_structured.csv",
    top_k:           int = 5,
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
        comp_struct["training_files"]+[comp_struct["submission_file"]]
    )
    

    # 3) profile each one
    all_schemas = {}
    for p in downloaded_paths:
        prof = describe_schema(
        source_path=str(p),
        target_column=comp_struct["target_column"]
        )
        # 2) collapse both the schema and the target_summary if they are too big
        compacted = compact_profile_for_llm(
            prof,
            max_features=MAX_FEATURES,    # collapse runs if > n features
            max_classes=MAX_CLASSES      # keep top n classes per target
        )

        all_schemas[p.name] = compacted

    temp_dir, tabular_path = extract_tabular(f"test/{slug}/{slug}/{comp_struct["submission_file"]}")
    sep = '\t' if tabular_path.lower().endswith('.tsv') else ','

    try:
        df = pd.read_csv(tabular_path, sep=sep, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(tabular_path, sep=sep, encoding='latin1')

    submission_style = df.head(0).to_dict(orient='records')

    print("Sample submission columns:", df.columns.tolist())

    comp_struct["data_profiles"] = all_schemas

    range_specs = None
    if len(comp_struct["target_column"]) <= MAX_CLASSES:    
        range_specs = compact(comp_struct) 
        comp_struct["target_column_ranges"] = range_specs

    desc_path = Path(f"{comp_folder}/{slug}_desc.json")
    desc_path.write_text(json.dumps(comp_struct, ensure_ascii=False, indent=2), encoding="utf-8")  

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
        "competition_problem_description":  comp_struct["competition_problem_description"],
        "competition_problem_subtype":      comp_struct["competition_problem_subtype"],
        "dataset_metadata":                 comp_struct["dataset_metadata"],
        "data_profiles":                    comp_struct["data_profiles"],
        "files_preprocessing_instructions": comp_struct["files_preprocessing_instructions"],
        "submission_example":               submission_style,
        "files_list":                       comp_struct["files_list"],
        "examples": [
            {"score": sc, "preprocessing_steps": prep, "model_layers_code": layers}
            for (_, _, sc, prep, layers) in examples
        ]
        
    }

    # 1) Initial Responses API call
    system_msg = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer and data scientist. "
            "Do not emit plain text. Populate *only* the `notebook_code` argument "
            "(no other keys)."
        )
    }
    user_msg   = { "role":"user",   "content": json.dumps(payload, ensure_ascii=False) }

    response = openai.responses.create(
        model="o4-mini",
        input=[system_msg, user_msg],
        tools=tools,
        store=True
    )

    func = next(
        ev for ev in response.output
        if ev.type == "function_call"
        and ev.name == "generate_keras_schema"
    )
    args = json.loads(func.arguments)


    args.pop("notebook_code", None)
    # run your local impl
    notebook = generate_keras_schema_impl(**args)

    # wrapup and get final
    final = openai.responses.create(
        model=OPENAI_MODEL,
        input=response.output + [{
            "type":    "function_call_output",
            "call_id": func.call_id,
            "output":  notebook
        }],
        tools=tools
    )

    return postprocess_code(final.output_text)

def solve_competition_tuner(slug: str) -> str:
    base = Path(f"test/{slug}")

    comp_struct = json.loads((base / f"{slug}_desc.json").read_text(encoding="utf-8"))
    # load the existing keras solution
    existing_solution_code = (base / f"{slug}_solution.py").read_text(encoding="utf-8")

    # 1) Select the best‐matching profile key:
    profile_key = select_hyperparameter_profile(comp_struct, HYPERPARAMETER_BANK)
    # 2) Pull out that single profile dict:
    chosen_profile = HYPERPARAMETER_BANK[profile_key]
    print(chosen_profile)
    hp_payload = chosen_profile


    payload = {
        "competition_slug":                slug,
        "competition_problem_description": comp_struct["competition_problem_description"],
        "competition_problem_type":        comp_struct["competition_problem_type"],
        "competition_problem_subtype":     comp_struct["competition_problem_subtype"],
        "dataset_metadata":                comp_struct["dataset_metadata"],
        "data_profiles":                   comp_struct["data_profiles"],
        "existing_solution_code":          existing_solution_code,
        "hyperparameter_bank":             hp_payload
    }

    range_specs = compact(comp_struct) 

    if len(comp_struct["target_column"]) <= MAX_CLASSES:
        payload["target_columns"] = comp_struct["target_column"]
    else:
        payload["target_column_ranges"] = range_specs

    system_msg = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer.  "
            "Use the `hyperparameter_bank` to pick or combine the most appropriate hyperparameters for keras tuner"
            "for this competition’s data and emit ONLY valid JSON via the function schema."
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
        functions=[generate_tuner_schema],
        function_call={"name": "generate_tuner_schema"}
    )

    msg       = response.choices[0].message
    args      = json.loads(msg.function_call.arguments)
    tuner_code = args["tuner_code"]
    
    return tuner_code


"""
    Follow-up prompt
"""

def followup_prompt(
    slug: str,
    kt: bool
) -> str:

    solution_path = f"test/{slug}/{slug}{'_kt' if kt else ''}_solution.py"

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
            "Now you will be given a deep learning <Code> along with the <Error> log. Think step by step and generate a fix for this code, but only fix the issue mentioned, do not modify anything else. Rewrite the full code from the begining, fixing the bug. In you code, include the code that records the time of how long the model trains. Write the code in this format"
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




