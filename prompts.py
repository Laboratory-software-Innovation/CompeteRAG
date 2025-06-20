import json
import re
import pandas as pd
from pathlib import Path

import openai

from config import OPENAI_MODEL,kaggle_api
from selenium_helper import init_selenium_driver
from utils import fetch_competition_page_html, parse_competition_metadata, describe_schema
from similarity import find_similar_ids

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



"""
    Initial prompt for Keras and Keras-Tuner  
"""

def solve_competition_with_code(
    class_col:       str,
    slug:            str, 
    structured_csv:  str = "notebooks_structured.csv",
    top_k:           int = 5,
    kt:              bool = 1, 
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


