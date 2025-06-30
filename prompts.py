label_competition_schema = {
    "name": "label_competition_schema",
    "description": (
        "Given:\n"
        "  - the raw competition metadata and dataset metadata,\n"
        "  - files_list - files that were parsed and that are available for download:\n"
        "  - all_files: an array of all files retrieved using kaggle api, may not contain all due to limitations\n"
        "Retrieve:\n"
        "  - training_files: Based on dataset_metadata, files_list give [<string>, …],  an array of all training tabular files that need to be downloaded\n"
        "  - target_column: an array of all column names in the dataset that must be predicted\n"
        "  - submission_file: a string containing an exact name of a file that can be used a submission example"
        "Emit ONLY these two fields as JSON—no extra keys, no prose, no markdown."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_column": {
                "type": "array",
                "items": {"type": "string"},
                "description": "an array of all column names in the dataset that must be predicted"
            },
            "files_list": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Raw list of files from the Kaggle data tab"
            },
            "all_files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "an array of all files retrieved using kaggle api, may not contain all due to limitations"
            },
            "training_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Based on dataset_metadata give [<string>, …],  an array of all training tabular files that need to be downloaded."
            },
            "submission_file": {
                "type": "string",
                "description": "Based on dataset_metadata give a string containing the exact name of the file that can be used as a submission example."
            }
        },
        "required": ["target_column", "training_files","submission_file"]
    }
}



# collection ---> collect_and_structure
ask_structured_schema = {
    "name": "ask_structured_schema",
    "description": (
        "**IMPORTANT**: Your *entire* response must be valid JSON matching this schema—**no** single-quotes, no Python `None`, no trailing commas, no code fences,\n"
        "From the competition metadata, dataset metadata, and a raw Jupyter notebook text, "
        "extract exactly these fields as JSON (no extra keys, no prose, no markdown):\n"
        "  - competition_problem_type: one of ['classification','regression']\n"
        "  - competition_problem_subtype: single, concise, lowercase‐and‐hyphenated phrase (e.g. “binary classification”, “multiclass classification”, “multi-label classification”, “time-series forecasting”, “continuous regression”, “ordinal regression”, etc. or any other that fits.)\n"
        "  - competition_problem_description: dense, short, factual description of the problem, what needs to be found, no repetitive words (omit dataset‐meta here)\n"
        "  - dataset_metadata: plain‐English dataset_metadata in plain English as a single coherent paragraph, removing any non-human symbols (no bullets or symbols)\n"
        "  - competition_dataset_type: one of ['Tabular','Time-series','Text','Image','Audio','Video','Geospatial','Graph','Multimodal']\n"
        "  - preprocessing_steps: array of strings, each describing one transformation (e.g. 'median‐impute missing values')\n"
        "  - notebook_model_layers_code: literal code snippet that builds(e.g model.fit) each layer(e.g Dense, Conv, etc..) and compiles the model(e.g model.compile) \n"
        "  - used_technique: either 'DL' or 'ML'\n"
        "  - library: string naming the main library used (exactly one 'Tensorflow', 'Pytorch')\n"
        "  - target_column: array of all column names in the dataset that must be predicted \n"
        "Emit ONLY those keys."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "competition_problem_type": {
                "type": "string",
                "enum": ["classification", "regression"],
                "description": "Pick exactly one."
            },
            "competition_problem_subtype": {
                "type": "string",
                "description": "Single lowercase hyphenated phrase describing the subtype."
            },
            "competition_problem_description": {
                "type": "string",
                "description": "Dense, factual description of what needs to be predicted."
            },
            "dataset_metadata": {
                "type": "string",
                "description": "Plain‐English paragraph describing the dataset."
            },
            "competition_dataset_type": {
                "type": "string",
                "enum": ["Tabular","Time-series","Text","Image","Audio","Video","Geospatial","Graph","Multimodal"],
                "description": "Choose one primary modality."
            },
            "preprocessing_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List every transformation (scaling, normalization, one-hot encoding, etc.) in plain English."
            },
            "notebook_model_layers_code": {
                "type": "string",
                "description": (
                    "include the literal code lines of model compile, model fit, and that build each layer (e.g. `Dense(128, activation='relu', …)`)\n"
                    "The line(s) that create or instantiate the model (Sequential, Functional, subclass, torch.nn.Module, etc.).\n"
                    "All layer-construction calls (Dense, Conv2D, custom layers, etc.) or layer definitions in a subclass.\n"
                    "The call that compiles or configures training (e.g. `compile()`, `configure_optimizers()`, etc.).\n"
                    "The call that launches training (e.g. `fit()`, `trainer.fit()`, `train()`, etc.).\n"
                    "Do not include unrelated code, helper wrappers, or omit any of these steps."
                )
            },
            "used_technique": {
                "type": "string",
                "enum": ["DL","ML"],
                "description": "Either 'DL' or 'ML'."
            },
            "library": {
                "type": "string",
                "description": "Name of the main library used."
            },
            "target_column": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of all column names in train.csv to predict."
            }
        },
        "required": [
            "competition_problem_type",
            "competition_problem_subtype",
            "competition_problem_description",
            "dataset_metadata",
            "competition_dataset_type",
            "preprocessing_steps",
            "notebook_model_layers_code",
            "used_technique",
            "library",
            "target_column"
        ]
    }
}

# llm_coding ---> structure_and_label_competition
structure_and_label_competition_schema = {
    "name": "structure_and_label_competition_schema",
    "description": (
        "Given raw Kaggle competition metadata, dataset metadata and a list of files, "
        "return exactly the following fields as JSON:\n"
        "  - competition_problem_type (\"regression\" or \"classification\")\n"
        "  - competition_problem_subtype (lower-case, hyphenated phrase describing the subtype)\n"
        "  - competition_problem_description (dense, non-repetitive description of the goal)\n"
        "  - dataset_metadata (plain-English paragraph rewrite of the original)\n"
        "  - competition_dataset_type (one of: Tabular, Time-series, Text, Image, Audio, Video, Geospatial, Graph, Multimodal)\n"
        "  - target_column (array of the exact label column name(s) in the training files)\n"
        "  - files_list (the raw file names discovered on the data tab)\n"
        "  - all_files - All files used for the competition available for download, may not include all of them\n"
        "  - training_files (subset of files_list to load as training tables)\n"
        "  - files_preprocessing_instructions (plain-English instructions to prep those files)\n"
        "No extra keys, no prose—just that JSON object."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "competition_problem_type": {
                "type": "string",
                "description": "“regression” or “classification”"
            },
            "competition_problem_subtype": {
                "type": "string",
                "description": "Specific problem subtype, lower-case and hyphenated"
            },
            "competition_problem_description": {
                "type": "string",
                "description": "Dense description of what needs to be predicted"
            },
            "dataset_metadata": {
                "type": "string",
                "description": "Rewritten dataset_metadata in plain English"
            },
            "competition_dataset_type": {
                "type": "string",
                "enum": ["Tabular","Time-series","Text","Image","Audio","Video","Geospatial","Graph","Multimodal"],
                "description": "Primary data modality"
            },
            "target_column": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Label column(s) that must be predicted"
            },
            "files_list": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Raw list of files from the Kaggle data tab"
            },
            "all_files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "All files available for download"
            },
            "training_files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "**Based on the files_list, all_files, and dataset_metadata, give  an array of exact names of all training tabular files that need to be downloaded, ensure that the listed files in the dataset_metadata correspond to the ones in files_list, if not go with the file most similar in the files_list"
            },
            "files_preprocessing_instructions": {
                "type": "string",
                "description": "Based on the dataset_metadata and the files observed, write an instruction on how to preprocess(drop features, split the dataset if no testing was given etc, etc)"
            }
        },
        "required": [
            "competition_problem_type",
            "competition_problem_subtype",
            "competition_problem_description",
            "dataset_metadata",
            "competition_dataset_type",
            "target_column",
            "files_list",
            "all_files",
            "training_files",
            "files_preprocessing_instructions"
        ]
    }
}



# llm_coding ----> solve_competition_keras
generate_keras_schema = {
        "name": "generate_keras_schema",   
        "description": (
            "***Generate and save a runnable Python code wrapped in <Code>…</Code> in the `notebook_code` json field:\n"
            "  - Every backslash (`\\`) in your code must be escaped as `\\\\`, every double‐quote (`\"`) inside your code must be escaped as `\\\"`, and every newline as `\\n`\n"
            "  - Ensure that the code you emit is **syntactically valid Python**:\n"  
            "  - Every `import` must be on its own line. \n "
            "  - Use consistent 4-space indentation for all blocks.\n"  
            "  - All parentheses `()`, brackets `[]` and braces `{}` must be balanced and preserved.\n"
            "Given:\n"
            "  - `competition_slug`: the Kaggle competition slug,\n"
            "  - `competition_problem_description`: Dense competition problem description,\n"
            "  - `competition_problem_type`: Classification|Regression,\n"
            "  - `competition_problem_description`: Specifies the subtype of the problem,\n"
            "  - `dataset_metadata`: Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided,\n"
            "  - `data_profiles`: compacted schema & target summaries for each file,\n"
            "  - `files_preprocessing_instructions`: suggested data–prep steps,\n"
            "  - `target_columns`: (optional) List of one or more exact column names to predict, if there are only a few targets.\n"
            "  - `target_column_ranges`: (optional) A compact array of `{prefix, min_index, max_index}` objects used when there are many targets sharing numeric suffixes.  Your code should reconstruct:\n"
            "      ```python\n"
            "      target_cols = []\n"
            "      for spec in target_column_ranges:\n"
            "          p, lo, hi = spec[\"prefix\"], spec[\"min_index\"], spec[\"max_index\"]\n"
            "          target_cols = [f\"{p}{i}\" for i in range(lo, hi1)]\n"
            "      ```\n"
            "  - `training_files`: list of one or more CSV/TSV files to read,\n"
            "  - `all_files`: list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset,\n"        
            "  - `examples`: top-K example kernels for inspiration,\n"
            "The generated code must implement:\n"
            "0. **Target encoding**: after reading the dataset and before any split,\n"
            "   use `LabelEncoder().fit(y)` → `y_enc = le.transform(y)` and keep `le.classes_` for later.\n"
            "   `target_columns`: if length==1, use `LabelEncoder` on that column;"
            "   if length>1, build `Y = df[target_columns].values` (and skip `stratify`)"
            "1. **Reproducibility**: set seeds for Python, NumPy, scikit-learn, and TensorFlow (or PyTorch).\n"
            "2. **Imports**: `pandas`, `numpy`,\n"
            "   - `sklearn.model_selection.train_test_split`,\n"
            "   - `sklearn.impute.SimpleImputer`,\n"
            "   - `sklearn.compose.ColumnTransformer`,\n"
            "   - `sklearn.preprocessing.StandardScaler`,`OneHotEncoder`,\n"
            "   - `sklearn.pipeline.Pipeline`,\n"
            "   - `tensorflow` (or `torch`),\n"
            "   - `tensorflow.keras.callbacks.EarlyStopping,ModelCheckpoint` (or torch equivalents),\n"
            "   - `json`, `time`.\n"
            "3. **Data Loading**:\n"
            "   - Read every file in `training_files`.\n"
            "   - **Infer `target_cols` programmatically**: if you passed a common prefix (e.g. `start_`) or know the count `N`, generate your list in code instead of hard-coding.\n"
            "     **Special case:** If `target_columns` has more than 50 columns *and* they all share the same prefix P, you *must*:\n"
            "       ```python\n"
            "       prefix = \"P\"\n"
            "       N = <the maximum index>\n"
            "       target_cols = [f\"{prefix}{i}\" for i in range(1, N+1)]\n"
            "       ```\n"
            "     Do not inline the full list.\n""   "
            "**Safe column handling**:\n"
            "    1. Extract IDs and targets exactly by their raw CSV names, using a conditional pop/drop:\n"
            "    3. Proceed without ever renaming or lowercasing — all subsequent code should refer to columns exactly as in the original files schema.\n"
            "   - If there is more than one file, decide which ones are for training and testing based on `training_files` and `all_files` and `dataset_metadata` else split the single dataset 80/20 stratified on target.\n"
            "   - Detect & preserve any ID column.\n"
            "Preprocessing MUST ALSO include:\n"
            "  - Label encoding of the target column to integers using\n"
            "    `sklearn.preprocessing.LabelEncoder`, preserving the mapping for inverse transform.\n"
            "4. **Feature Engineering**: automatically drop or transform obviously irrelevant columns (e.g. all-nan, high-cardinalities), at your discretion.\n"
            "5. **Train/Validation Split**: if not already split"
            "6. **Preprocessing Pipeline**:\n"
            "   - Auto-detect numeric vs categorical via `df.select_dtypes`.\n"
            "   - Build `ColumnTransformer`:\n"
            "       - Numeric: `SimpleImputer(strategy='median', add_indicator=True)` → `StandardScaler()`\n"
            "       - Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`\n"
            "   - Fit-transform train and transform val/test.\n"
            "7. **Determine feature dimension**: `input_shape = X_train_proc.shape[1]`.\n"
            "8. **Model Definition**: build an ANN in Keras/TensorFlow (or PyTorch) with at least two hidden layers, including `Dropout` or `BatchNormalization`.\n"
            "9. **Compilation**: `Adam` optimizer, `binary_crossentropy` (or `mse`), metrics `['accuracy']` (or `['RootMeanSquaredError']`).\n"
            "10. **Callbacks & Training**: `EarlyStopping(monitor='val_loss', patience=5)`   `ModelCheckpoint(save_best_only=True)`, up to 100 epochs, record training duration.\n"
            "11. **Evaluation & Logging**: load best weights, extract `training_accuracy`, `training_loss`, `validation_accuracy`, `validation_loss`, save to `results.json`.\n"
            "12. **Prediction & Submission**: transform test set, predict, and write 'submission_result.csv' with preserved IDs as first column.\n"
            "    - For binary or multilabel classification, threshold the output probabilities at 0.5 and cast to int, so all target columns are 0 or 1 (no floats).\n"
            "    - For multiclass (single target), use argmax for class labels and write integer class indices or decoded labels as expected.\n"
            "    - For regression, output the predicted values as-is (do not round or threshold).\n"
            "14. Load the test data by reading the testing files into a DataFrame.\n"
            "15. Extract the ID column (id_col) and save it as ids_test.\n"
            "16. Remove any remaining target columns so only feature columns remain.\n"
            "17. Apply the preprocessing pipeline (preprocessor) to transform the feature DataFrame.\n"
            "18. Use the model to predict on the preprocessed test features and store the predictions.\n"
            "19. Create the submission DataFrame from the predictions, naming columns after the targets.\n"
            "20. Insert ids_test as the first column of the submission DataFrame.\n"
            "21. Write the submission DataFrame to 'submission_result.csv' without row indices."
            "22. **Always include the test prediction and submission code at the end, loading 'test.csv', predicting with 'best_model', and saving to 'submission_result.csv'. Use 'id' and the correct target column(s)."
            "23. **If there are multiple target columns, save each column in the submission DataFrame as required.\n"
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "competition_slug": {
                    "type": "string",
                    "description": "The Kaggle competition slug."
                },
                "competition_problem_description": {
                    "type": "string",
                    "description": "Dense competition description giving the core goal."
                },
                "competition_problem_type": {
                    "type": "string",
                    "description": "Classification|Regression"
                },
                "competition_problem_subtype": {
                    "type": "string",
                    "description": "Specifies the subtype of the problem"
                },
                "dataset_metadata": {
                    "type": "string",
                    "description": "Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided"
                },
                "data_profiles": {
                    "type": "object",
                    "description": (
                        "A mapping filename → compacted schema & target summary, "
                        "as returned by compact_profile_for_llm."
                    )
                },
                "files_preprocessing_instructions": {
                    "type": "string",
                    "description": "Instructions for how to preprocess the raw files."
                },
                "target_columns": {
                "type": "array",
                "description": "If there are many targets, group them by prefix+index ranges instead of listing each.",
                "items": {
                    "type": "object",
                    "properties": {
                    "prefix":     { "type": "string",  "description": "Common prefix of these target columns, e.g. 'start.'"},
                    "min_index":  { "type": "integer", "description": "Lowest numeric suffix seen"},
                    "max_index":  { "type": "integer", "description": "Highest numeric suffix seen"},
                    "count":      { "type": "integer", "description": "How many columns in this range"}
                    },
                    "required": ["prefix","min_index","max_index"]
                }
                },
                "training_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of all training‐set filenames to read."
                },
                "all_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of all files include in the competition, including training and testing files"
                },
                "examples": {
                    "type": "array",
                    "description": "Top‐K similar kernels for inspiration.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kernel_ref":           {"type":"string"},
                            "score":                {"type":"number"},
                            "preprocessing_steps":  {
                                "type":"array",
                                "items":{"type":"string"}
                            },
                            "model_layers_code":    {"type":"string"}
                        },
                        "required":["kernel_ref","score","preprocessing_steps","model_layers_code"]
                    }
                },
                "notebook_code": {
                    "type": "string",
                    "description": "***The complete runnable Python code wrapped in <Code>…</Code>, saved as an output.***"
                }
            },
            "required": [
                "notebook_code"
            ]
        }
    }



generate_tuner_schema = {
    "name": "generate_tuner_schema",
    "description": (
        "***Generate and save a runnable Python code wrapped in <Code>…</Code> in the ***`tuner_code` json field"
            "  - Every backslash (`\\`) in your code must be escaped as `\\\\`, every double‐quote (`\"`) inside your code must be escaped as `\\\"`, and every newline as `\\n`\n"
            "  - Ensure that the code you emit is **syntactically valid Python**:\n"  
            "  - Every `import` must be on its own line. \n "
            "  - Use consistent 4-space indentation for all blocks.\n"  
            "  - All parentheses `()`, brackets `[]` and braces `{}` must be balanced and preserved.\n"  
        "Given:\n"
            "  - `competition_slug`: the Kaggle competition slug,\n"
            "  - `competition_problem_description`: Dense competition problem description,\n"
            "  - `competition_problem_type`: Classification|Regression,\n"
            "  - `competition_problem_description`: Specifies the subtype of the problem,\n"
            "  - `dataset_metadata`: Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided,\n"
            "  - `data_profiles`: compacted schema & target summaries for each file,\n"
            "  - `existing_solution_code`: the text of the working Keras solution,\n"      
            "  - `hyperparameter_bank`: an object containing a predefined profile for this subtype of model\n"   
            "  - Emit ONLY a single JSON object with exactly one field: "
            "  - ***`tuner_code`: a string containing the **full** runnable Python notebook code wrapped in `<Code>…</Code>`\n"
            "  - This must include **all** original data loading, preprocessing, model definition, callbacks, training, **and** the Keras-Tuner integration (imports, HyperModel wrapper, tuner setup, search, and best_model rebuild), as well as final evaluation and `submission.to_csv`.\n"        "    (including imports, HyperModel subclass, tuner setup, search, and final retrain)\n"
            "  - Keep the structure the same as the original Keras code, including the training timing, saving the result into a submission file \n"
        "   0.1. Use `chosen_profile[\"params\"]` to drive every `hp.*` call below.\n" 
        "***IMPORTANT CODING RULES:***\n"
        "  - First, select `hyperparameter_bank` by comparing each bank-entry’s `tags` to `competition_problem_type`, `competition_problem_subtype`, and whether the data is tabular/text/image.\n"
        "  - You must only use hyperparameters defined in `hyperparameter_bank.params`.  For each key in `hyperparameter_bank.params`, generate exactly one `hp.*(...)` call _inside_ `build(self, hp)`.\n"
        "  - Map each param name to its correct `hp.Int` / `hp.Float` / `hp.Choice` signature, preserving min/max/step/log/values exactly.\n"
        "  - NEVER call `hp.*` or `kt.HyperParameters()` anywhere else in the code.\n"
        "  - In `tuner.search(...)`, pass **literal** `batch_size` and `epochs` values drawn from `chosen_profile.params.batch_size` and `.epochs`, but **do not** call `hp.*` there.\n"
        "  - Do not introduce any new hyperparameters or omit any from `chosen_profile.params`.\n"
        "  - All variable names (e.g. `X_train_proc`, `early_stopping`, etc.) must match the original Keras code exactly.\n"
        "  - The final generated code must be valid Python and runnable end‐to‐end with no missing variables or undefined names.\n"
        "   - DO NOT rename variables—match the exact names used in the provided Keras code.\n"
        "  - This guarantees that `hp` is only referenced inside `build(self, hp)` and never in the `search` call.\n"
        "\n"
        "# TUNER INTEGRATION (replace original build & fit):\n"
        "1. Add `import keras_tuner as kt` alongside existing imports.\n"
        "2. Wrap your existing `build_model(...)` in:\n"
        "   ```python\n"
        "   class MyHyperModel(kt.HyperModel):\n"
        "       def build(self, hp):\n"
        "           model = build_model(\n"
        "               ... # hyperparameters\n"
        "           )\n"
        "           # Compile the model here using the same optimizer/loss/metrics as the original Keras code\n"
        "           model.compile(optimizer=..., loss=..., metrics=[...])\n"
        "           return model\n"
        "   ```\n"
        "3. Instead of any `model.fit(...)`, insert:\n"
        "   ```python\n"
        "    **Use placeholders** for:\n"
        "     - `max_trials` ← `chosen_profile.params.max_trials` if present, otherwise a sensible default (e.g. 20).\n"
        "     - `batch_size` ← the midpoint (or any deterministic pick) of `chosen_profile.params.batch_size.values`.\n"
        "     - `epochs` ← the midpoint (or pick) of `chosen_profile.params.epochs.min` and `.max`.\n"
        "  4. In your `tuner.search`, use these placeholders literally:\n"
        "     ```python\n"
        "     max_trials = <PLACEHOLDER_max_trials>\n"
        "     batch_size = <PLACEHOLDER_batch_size>\n"
        "     epochs     = <PLACEHOLDER_epochs>\n\n"
        "     tuner = kt.RandomSearch(\n"
        "         MyHyperModel(),\n"
        "         objective='val_loss',\n"
        "         max_trials=max_trials,\n"
        "         executions_per_trial=1,\n"
        "         seed=42\n"
        "     )\n\n"
        "     tuner.search(\n"
        "         X_train_proc, Y_train,\n"
        "         validation_data=(X_val_proc, Y_val),\n"
        "         batch_size=batch_size,\n"
        "         epochs=epochs,\n"
        "         callbacks=[early_stopping, checkpoint]\n"
        "     )\n"
        "     ```\n"
        "  5. Replace `<PLACEHOLDER_…>` with values computed from your `chosen_profile.params`.\n"
        "\n"
        "best_hp = tuner.get_best_hyperparameters(1)[0]\n"
        "best_model = tuner.hypermodel.build(best_hp)\n"
        "   ```\n"
        ""
        "4. Hand back `best_model` into your existing evaluation & submission code.\n"
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "competition_slug":                 {"type": "string"},
            "competition_problem_description":  {"type": "string"},
            "competition_problem_type":         {"type": "string"},
            "competition_problem_subtype":      {"type": "string"},
            "dataset_metadata":                 {"type": "string"},
            "data_profiles":                    {"type": "object"},
            "existing_solution_code":           {"type": "string"},
            "hyperparameter_bank": {
                "type": "object",
                "description": "A map from profile name → hyperparameter profile.  Each profile has `tags`, `description`, `params`, `advice`, and `source`.",
                "additionalProperties": {
                    "type": "object",
                    "required": ["tags","description","params"],
                    "properties": {
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of tags describing this profile (e.g. classification, tabular, low_features)."
                    },
                    "description": { "type": "string" },
                    "params": {
                        "type": "object",
                        "description": "Each key is a hyperparameter name; the value describes its type and bounds.",
                        "additionalProperties": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": { "type": "string", "enum": ["int","float","choice","boolean"] },
                            "min":    { "type": "number" },
                            "max":    { "type": "number" },
                            "step":   { "type": "number" },
                            "values": { "type": "array", "items": {} },
                            "sampling": { "type": "string", "enum": ["linear","log"] }
                        }
                        }
                    },
                    "advice": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "source": { "type": "string" }
                    }
                }
            },

            "tuner_code": {
                "type": "string",
                "description": (
                  "***The complete runnable Python code for the competition notebook, including data loading, preprocessing pipeline, original model definition, "
                  "callbacks, Keras-Tuner integration (imports, HyperModel subclass, tuner setup & search), final best-model training/evaluation, and submission code, "
                  "all wrapped in `<Code>…</Code>`, saved as an output.***"
                )
            }
        },
        "required": [
            "tuner_code"
        ]
    }
}






"""
generate_keras_schema = {
        "name": "generate_keras_schema",   
        "description": (
            "***Based on the given information you should save a string containing **only** runnable Python code wrapped in `<Code>…</Code>"
            "Given:\n"
            "  - `competition_slug`: the Kaggle competition slug,\n"
            "  - `competition_problem_description`: Dense competition problem description,\n"
            "  - `competition_problem_type`: Classification|Regression,\n"
            "  - `competition_problem_description`: Specifies the subtype of the problem,\n"
            "  - `dataset_metadata`: Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided,\n"
            "  - `data_profiles`: compacted schema & target summaries for each file,\n"
            "  - `files_preprocessing_instructions`: suggested data–prep steps,\n"
            "  - `target_columns`: List of one or more columns to predict (for multi-output tasks)."
            "  - `training_files`: list of one or more CSV/TSV files to read,\n"
            "  - `all_files`: list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset,\n"        
            "  - `examples`: top-K example kernels for inspiration,\n"
            "The generated code must implement:\n"
            "0. **Target encoding**: after reading the dataset and before any split,\n"
            "   use `LabelEncoder().fit(y)` → `y_enc = le.transform(y)` and keep `le.classes_` for later.\n"
            "   `target_columns`: if length==1, use `LabelEncoder` on that column;"
            "   if length>1, build `Y = df[target_columns].values` (and skip `stratify`)"
            "1. **Reproducibility**: set seeds for Python, NumPy, scikit-learn, and TensorFlow (or PyTorch).\n"
            "2. **Imports**: `pandas`, `numpy`,\n"
            "   - `sklearn.model_selection.train_test_split`,\n"
            "   - `sklearn.impute.SimpleImputer`,\n"
            "   - `sklearn.compose.ColumnTransformer`,\n"
            "   - `sklearn.preprocessing.StandardScaler`,`OneHotEncoder`,\n"
            "   - `sklearn.pipeline.Pipeline`,\n"
            "   - `tensorflow` (or `torch`),\n"
            "   - `tensorflow.keras.callbacks.EarlyStopping,ModelCheckpoint` (or torch equivalents),\n"
            "   - `json`, `time`.\n"
            "3. **Data Loading**:\n"
            "   - Read every file in `training_files`.\n"
            "   - **Infer `target_cols` programmatically**: if you passed a common prefix (e.g. `start_`) or know the count `N`, generate your list in code instead of hard-coding. For example:\n"
            "       ```python\n"
            "       # if all target columns begin with the same prefix:\n"
            "       prefix = target_columns_prefix  # e.g. 'start_'\n"
            "       target_cols = [c for c in df.columns if c.startswith(prefix)]\n"
            "       # or if you know exactly N targets:\n"
            "       target_cols = [f\"{prefix}{i}\" for i in range(N)]\n"
            "       ```\n"
            "   **Safe column handling**:"
            "    1. Extract IDs and targets exactly by their raw CSV names, using a conditional pop/drop:"
            "    ```python\n"
            "    id_col     = 'id'\n"
            "    target_cols = target_columns   \n"
            "    ids        = df.pop(id_col)\n"
            "    if len(target_cols) == 1:\n"
            "        # single-output\n"
            "        Y = df.pop(target_cols[0])\n"
            "    else:\n"
            "        # multi-output: slice and then drop\n"
            "        Y = df[target_cols]\n"
            "        df.drop(columns=target_cols, errors='ignore', inplace=True)\n"
            "    ```\n"
            "    2. Drop any other non-feature columns by name (no renaming ever):\n"
            "        ```python\n"
            "        X = df.drop(columns=target_cols   [id_col], errors='ignore')\n"
            "        ```\n"
            "    3. Proceed without ever renaming or lowercasing — all subsequent code should refer to columns exactly as in the original files.\n"
            "   - If there is more than one file, decide which ones are for training and testing based on `training_files` and `all_files` and `dataset_metadata` else split the single dataset 80/20 stratified on target.\n"
            "   - Detect & preserve any ID column.\n"
            "Preprocessing MUST ALSO include:\n"
            "  - Label encoding of the target column to integers using\n"
            "    `sklearn.preprocessing.LabelEncoder`, preserving the mapping for inverse transform.\n"
            "4. **Feature Engineering**: automatically drop or transform obviously irrelevant columns (e.g. all-nan, high-cardinalities), at your discretion.\n"
            "5. **Train/Validation Split**: if not already split:\n"
            "     - If `Y.ndim == 1` (single-output), call\n"
            "         ```python\n"
            "         X_train, X_val, Y_train, Y_val = train_test_split(\n"
            "             X, Y, test_size=test_size, random_state=seed, stratify=Y\n"
            "         )\n"
            "         ```\n"
            "     - Else (multi-output), call without `stratify`:\n"
            "         ```python\n"
            "         X_train, X_val, Y_train, Y_val = train_test_split(\n"
            "             X, Y, test_size=test_size, random_state=seed\n"
            "         )\n"
            "         ```\n"
            "6. **Preprocessing Pipeline**:\n"
            "   - Auto-detect numeric vs categorical via `df.select_dtypes`.\n"
            "   - Build `ColumnTransformer`:\n"
            "       - Numeric: `SimpleImputer(strategy='median', add_indicator=True)` → `StandardScaler()`\n"
            "       - Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`\n"
            "   - Fit-transform train and transform val/test.\n"
            "7. **Determine feature dimension**: `input_shape = X_train_proc.shape[1]`.\n"
            "8. **Model Definition**: build an ANN in Keras/TensorFlow (or PyTorch) with at least two hidden layers, including `Dropout` or `BatchNormalization`.\n"
            "9. **Compilation**: `Adam` optimizer, `binary_crossentropy` (or `mse`), metrics `['accuracy']` (or `['RootMeanSquaredError']`).\n"
            "10. **Callbacks & Training**: `EarlyStopping(monitor='val_loss', patience=5)`   `ModelCheckpoint(save_best_only=True)`, up to 100 epochs, record training duration.\n"
            "11. **Evaluation & Logging**: load best weights, extract `training_accuracy`, `training_loss`, `validation_accuracy`, `validation_loss`, save to `results.json`.\n"
            "12. **Prediction & Submission**: transform test set, predict, threshold at 0.5 if classification, write `submission_result.csv` with preserved IDs .\n"
            "# Load test set\n"
            "test_df = pd.read_csv('test.csv')\n"
            "# Extract IDs\n"
            "ids_test = test_df.pop(id_col)\n"
            "# Drop any leftover target columns\n"
            "X_test = test_df.drop(columns=target_cols, errors='ignore')\n"
            "# Transform features\n"
            "X_test_proc = preprocessor.transform(X_test)\n"
            "# Predict\n"
            "predictions = model.predict(X_test_proc)\n"
            "submission = pd.DataFrame(predictions, columns=target_cols)\n"
            "submission.insert(0, id_col, ids_test)\n"
            "# **Write** submission file\n"
            "submission.to_csv('submission_result.csv', index=False)\n"
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "competition_slug": {
                    "type": "string",
                    "description": "The Kaggle competition slug."
                },
                "competition_problem_description": {
                    "type": "string",
                    "description": "Dense competition description giving the core goal."
                },
                "competition_problem_type": {
                    "type": "string",
                    "description": "Classification|Regression"
                },
                "competition_problem_subtype": {
                    "type": "string",
                    "description": "Specifies the subtype of the problem"
                },
                "dataset_metadata": {
                    "type": "string",
                    "description": "Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided"
                },
                "data_profiles": {
                    "type": "object",
                    "description": (
                        "A mapping filename → compacted schema & target summary, "
                        "as returned by compact_profile_for_llm."
                    )
                },
                "files_preprocessing_instructions": {
                    "type": "string",
                    "description": "Instructions for how to preprocess the raw files."
                },
                "target_columns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Exact column name(s) of the target feature(s), as they appear in the raw CSV."
                },
                "training_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of all training‐set filenames to read."
                },
                "all_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of all files include in the competition, including training and testing files"
                },
                "examples": {
                    "type": "array",
                    "description": "Top‐K similar kernels for inspiration.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kernel_ref":           {"type":"string"},
                            "score":                {"type":"number"},
                            "preprocessing_steps":  {
                                "type":"array",
                                "items":{"type":"string"}
                            },
                            "model_layers_code":    {"type":"string"}
                        },
                        "required":["kernel_ref","score","preprocessing_steps","model_layers_code"]
                    }
                },
                "notebook_code": {
                    "type": "string",
                    "description": "***The complete runnable Python code wrapped in <Code>…</Code>, saved as an output.***"
                }
            },
            "required": [
                "notebook_code"
            ]
        }
    }
"""