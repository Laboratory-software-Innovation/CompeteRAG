import json
import re
import pandas as pd
from pathlib import Path
from utils import extract_tabular, download_train_file

from typing import List, Dict, Optional
import openai

from collections import defaultdict
from config import OPENAI_MODEL,RESPONSES_MODEL,kaggle_api
from helpers.selenium_helper import init_selenium_driver
from utils import fetch_competition_page_html, parse_competition_metadata, parse_competition_data_tab, describe_schema, compact_profile_for_llm,select_hyperparameter_profile, compact_profiles
from similarity import find_similar_ids
from prompts import tools, tuner_tools, structure_and_label_competition_schema, extract_tools
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
        return code
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

        groups = defaultdict(list)
        for prefix, idx in parsed:
            groups[prefix].append(idx)

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

# ------------------------------
#             Keras
# ------------------------------
def generate_keras_schema_impl(tool_inputs: dict) -> str:

    print("INPUT ARGS:")
    print(json.dumps(tool_inputs, indent=2, ensure_ascii=False))
    print("submission_example:", tool_inputs.get("submission_example"))

    full_tool_spec = next(t for t in tools if t["name"] == "generate_keras_schema")
    keras_fn_schema = {
        "name":        full_tool_spec["name"],
        "description": full_tool_spec["description"],
        "parameters":  full_tool_spec["parameters"],
    }

    chat = openai.chat.completions.create(
        model          = OPENAI_MODEL,
        temperature    = 0.0,
        messages       = [
            {"role": "system",
             "content": "Generate only the notebook code wrapped in <Code>…</Code>."},
            {"role": "user",
             "content": json.dumps(tool_inputs, ensure_ascii=False)}
        ],
        functions      = [keras_fn_schema],
        function_call  = {"name": "generate_keras_schema"},
    )

    fc_args        = json.loads(chat.choices[0].message.function_call.arguments)
    notebook_code  = fc_args["notebook_code"]          
    return notebook_code


    
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
    print(comp_struct["competition_problem_subtype"])
    print("----------------")

    downloaded_paths = download_train_file(
        comp_meta["slug"],
        comp_folder,
        comp_struct["training_files"]+[comp_struct["submission_file"]]
    )
    

    all_schemas = {}
    for p in downloaded_paths:
        prof = describe_schema(
        source_path=str(p),
        target_column=comp_struct["target_column"]
        )
        compacted = compact_profile_for_llm(
            prof,
            max_features=MAX_FEATURES,  
            max_classes=MAX_CLASSES      
        )

        all_schemas[p.name] = compacted

    temp_dir, tabular_path = extract_tabular(f"test/{slug}/{slug}/{comp_struct["submission_file"]}")
    sep = '\t' if tabular_path.lower().endswith('.tsv') else ','

    try:
        df = pd.read_csv(tabular_path, sep=sep, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(tabular_path, sep=sep, encoding='latin1')

    first_row = df.iloc[0].to_dict()      
    submission_style  = [first_row]  

    print("Sample submission columns:",submission_style)

    comp_struct["data_profiles"] = all_schemas

    range_specs = None
    if len(comp_struct["target_column"]) <= MAX_CLASSES:    
        range_specs = compact(comp_struct) 
        comp_struct["target_column_ranges"] = range_specs

    desc_path = Path(f"{comp_folder}/{slug}_desc.json")
    desc_path.write_text(json.dumps(comp_struct, ensure_ascii=False, indent=2), encoding="utf-8")  

    df = pd.read_csv(structured_csv)
    df["kernel_ref_norm"] = df["kernel_ref"].apply(normalize_kernel_ref)

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


    comp_struct["data_profiles"] = compact_profiles(comp_struct["data_profiles"])


    payload = {
        "competition_problem_description":  comp_struct["competition_problem_description"],
        "competition_problem_subtype":      comp_struct["competition_problem_subtype"],
        "dataset_metadata":                 comp_struct["dataset_metadata"],
        "data_profiles":                    comp_struct["data_profiles"],
        "files_preprocessing_instructions": comp_struct["files_preprocessing_instructions"],
        "submission_example":               submission_style,
        "files_list":                       comp_struct["files_list"],
        "examples": [
            { "preprocessing_steps": prep, "model_layers_code": layers}
            for (_, _, _, prep, layers) in examples
        ]
        
    }   

    system_msg = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer and data scientist. "
            "Do not emit plain text. Populate *only* the notebook_code argument "
            "(no other keys)."
        )
    }
    user_msg = {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}

    plan = openai.responses.create(
        model  = RESPONSES_MODEL,
        input  = [system_msg, user_msg],
        tools  = tools,
        store  = True
    )

    func_evt   = next(ev for ev in plan.output
                      if ev.type == "function_call" and ev.name == "generate_keras_schema")
    tool_inputs = json.loads(func_evt.arguments)

    notebook_code = generate_keras_schema_impl(tool_inputs)

    openai.responses.create(
        model                = RESPONSES_MODEL,
        previous_response_id = plan.id,
        input = [{
            "type":    "function_call_output",
            "call_id": func_evt.call_id,
            "output":  notebook_code
        }],
        tools = tools,
        store = True,
    )
    return notebook_code

# ------------------------------
#          Keras-Tuner
# ------------------------------
#Helper to merge tuner snippet into our full Keras notebook
def merge_with_tuner(original_code: str, tuner_snippet: str) -> str:
    system_msg = {
        "role": "system",
        "content": "You are a precise Python refactoring assistant."
    }
    user_msg = {
        "role": "user",
        "content": (
            "Here is my full notebook:\n\n"
            "```python\n"
            f"{original_code}\n```\n\n"
            "And here is the new Keras-Tuner snippet (build, compile, search, retrain):\n\n"
            "```python\n"
            f"{tuner_snippet}\n```\n\n"
            "Please replace **only** the existing model-definition block—that is, **every line** \n"
            "from the first `model =` up to (but **not including**) the first `model.fit` call—with this Keras-Tuner snippet. \n"
            "**Keep** any variables it relies on (`n_features`, `n_classes`, `output_layer_original`, etc.) so it drops in cleanly, \n"
            "and **do not** touch imports, data loading, preprocessing, callbacks, logging, or the submission code. Return the full notebook text with only that block swapped out."
        )
    }
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[system_msg, user_msg]
    )
    return resp.choices[0].message.content






def generate_tuner_schema_impl(tool_inputs: dict) -> str:
    #Extract the original model block
    extract_resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": json.dumps({"original_code": tool_inputs["existing_solution_code"]})
        }],
        functions=extract_tools,
        function_call={"name": "extract_model_block"}
    )
    model_block = json.loads(extract_resp.choices[0].message.function_call.arguments)["model_block"]


    # grab the function spec
    full_spec = next(t for t in tuner_tools if t["name"]=="generate_tuner_schema")
    tuner_fn_schema = {
        "name":        full_spec["name"],
        "description": full_spec["description"],
        "parameters":  full_spec["parameters"],
    }
    #Generate tuner snippet from that block
    tuner_resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": json.dumps({
                "competition_problem_description": tool_inputs["competition_problem_description"],
                "competition_problem_subtype":     tool_inputs["competition_problem_subtype"],
                "model_block":                     model_block,
                "hyperparameter_bank":             tool_inputs["hyperparameter_bank"],
                "tuner_choice":                    tool_inputs["tuner_choice"],
            })
        }],
        functions=[tuner_fn_schema],
        function_call={"name":"generate_tuner_schema"}
    )
    return json.loads(tuner_resp.choices[0].message.function_call.arguments)["tuner_code"]


#Combine them in our solver
def solve_competition_tuner(slug: str) -> str:
    base          = Path(f"test/{slug}")
    comp_struct   = json.loads((base / f"{slug}_desc.json").read_text())
    existing_code = (base / f"{slug}_solution.py").read_text()

    profile_key  = select_hyperparameter_profile(comp_struct, HYPERPARAMETER_BANK)
    profile      = HYPERPARAMETER_BANK[profile_key]
    print(profile)
    tuner_choice = "bayesian" 

    tool_inputs = {
        "competition_problem_description": comp_struct["competition_problem_description"],
        "competition_problem_subtype":     comp_struct["competition_problem_subtype"],
        "existing_solution_code":          existing_code,
        "hyperparameter_bank":             profile,
        "tuner_choice":                    tuner_choice
    }

    # generate the tuner‐only snippet
    tuner_snippet = generate_tuner_schema_impl(tool_inputs)


    # merge back into the original notebook
    full_notebook = merge_with_tuner(existing_code, tuner_snippet)
    return full_notebook



#  Follow-up prompt
def followup_prompt(
    slug: str,
    kt: bool
) -> str:

    solution_path = f"test/{slug}/{slug}{'_kt' if kt else ''}_solution.py"

    path = Path(solution_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {solution_path}")
    
    text = path.read_text(encoding="utf-8")

    code_match = re.search(r"<Code>(.*?)</Code>", text, re.S)
    err_match  = re.search(r"<Error>(.*?)</Error>", text, re.S)
    if not code_match or not err_match:
        raise ValueError("File must contain <Code>...</Code> and <Error>...</Error> sections")

    code_block = code_match.group(1).strip()
    error_msg  = err_match.group(1).strip()

    base          = Path(f"test/{slug}")
    comp_struct = json.loads((base / f"{slug}_desc.json").read_text())

    extra = (
        f"<CompetitionProblemDescription>\n{comp_struct['competition_problem_description']}\n</CompetitionProblemDescription>\n"
        f"<CompetitionProblemSubtype>\n{comp_struct['competition_problem_subtype']}\n</CompetitionProblemSubtype>\n"
        f"<DatasetMetadata>\n{json.dumps(comp_struct['dataset_metadata'], indent=2)}\n</DatasetMetadata>\n"
        f"<DataProfiles>\n{json.dumps(comp_struct['data_profiles'], indent=2)}\n</DataProfiles>\n"
        f"<FilesPreprocessingInstructions>\n{comp_struct['files_preprocessing_instructions']}\n</FilesPreprocessingInstructions>\n"
        f"<FilesList>\n{json.dumps(comp_struct['files_list'], indent=2)}\n</FilesList>\n"
    )

    system = {
        "role": "system",
        "content": (
            "You are a world-class deep learning engineer with an expertice in debugging the code.  "
            "Turn on the verbose and save the training and validtion accuracy and log of the last epoch in a json file (results.json). It will have the following keys: {training_accuracy, training_loss,validation_accuracy and validation_loss}  "
            "Now you will be given a deep learning <Code> along with the <Error> log, and read‑only context wrapped in <CompetitionProblemDescription>, <CompetitionProblemSubtype>, <DatasetMetadata>, <DataProfiles>, <FilesPreprocessingInstructions>, <SubmissionExample>, and <FilesList>; use these sections as reference when diagnosing and fixing the bug, but do not modify or return them. "
            "Think step by step and generate a fix for this code, but only fix the issue mentioned, do not modify anything else. Rewrite the full code from the begining, fixing the bug. In you code, include the code that records the time of how long the model trains. Write the code in this format "
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
            f"{extra}"
            "Return only the corrected Python code, wrapped in <Code>...</Code>."
        )
    }


    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[system, user]
    )
    reply = resp.choices[0].message.content.strip()

    if reply.startswith("<Code>") and reply.endswith("</Code>"):
        return reply[len("<Code>"):-len("</Code>")].strip()
    return reply

