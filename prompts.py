label_competition_schema = {
    "name": "label_competition_schema",
    "description": (
        "From the raw competition and dataset metadata, extract exactly two fields:\n"
        "  • target_column: an array of all column names in the dataset that must be predicted\n"
        "  • files_list (the raw file names discovered on the data tab)\n"
        "  • training_files: Based on dataset_metadata give [<string>, …],  an array of all training tabular files that need to be downloaded\n"
        
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
            "training_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Based on dataset_metadata give [<string>, …],  an array of all training tabular files that need to be downloaded."
            }
            
        },
        "required": ["target_column", "files_list", "training_files"]
    }
}



# collection ---> collect_and_structure
ask_structured_schema = {
    "name": "ask_structured_schema",
    "description": (
        "**IMPORTANT**: Your *entire* response must be valid JSON matching this schema—**no** single-quotes, no Python `None`, no trailing commas, no code fences,\n"
        "From the competition metadata, dataset metadata, and a raw Jupyter notebook text, "
        "extract exactly these fields as JSON (no extra keys, no prose, no markdown):\n"
        "  • competition_problem_type: one of ['classification','regression']\n"
        "  • competition_problem_subtype: single, concise, lowercase‐and‐hyphenated phrase (e.g. “binary classification”, “multiclass classification”, “multi-label classification”, “time-series forecasting”, “continuous regression”, “ordinal regression”, etc. or any other that fits.)\n"
        "  • competition_problem_description: dense, short, factual description of the problem, what needs to be found, no repetitive words (omit dataset‐meta here)\n"
        "  • dataset_metadata: plain‐English dataset_metadata in plain English as a single coherent paragraph, removing any non-human symbols (no bullets or symbols)\n"
        "  • competition_dataset_type: one of ['Tabular','Time-series','Text','Image','Audio','Video','Geospatial','Graph','Multimodal']\n"
        "  • preprocessing_steps: array of strings, each describing one transformation (e.g. 'median‐impute missing values')\n"
        "  • notebook_model_layers_code: literal code snippet that builds(e.g model.fit) each layer(e.g Dense, Conv, etc..) and compiles the model(e.g model.compile) \n"
        "  • used_technique: either 'DL' or 'ML'\n"
        "  • library: string naming the main library used (exactly one 'Tensorflow', 'Pytorch')\n"
        "  • target_column: array of all column names in the dataset that must be predicted \n"
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


# llm_coding ----> solve_competition_with_code
generate_solution_schema = {
        "name": "generate_solution_schema",   
        "description": (
            "Given:\n"
            "  • `competition_slug`: the Kaggle competition slug,\n"
            "  • `competition_problem_description`: Dense competition problem description,\n"
            "  • `competition_problem_type`: Classification|Regression,\n"
            "  • `competition_problem_description`: Specifies the subtype of the problem,\n"
            "  • `dataset_metadata`: Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided,\n"
            "  • `data_profiles`: compacted schema & target summaries for each file,\n"
            "  • `files_preprocessing_instructions`: suggested data–prep steps,\n"
            "  • `target_columns`: List of one or more columns to predict (for multi-output tasks)."
            "  • `training_files`: list of one or more CSV/TSV files to read,\n"
            "  • `all_files`: list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset,\n"        
            "  • `examples`: top-K example kernels for inspiration,\n"
            "  • `use_kt`: boolean flag.\n\n"
            "Emit ONLY a single JSON object with exactly one field:\n"
            "  • ***`notebook_code`: a string containing **only** runnable Python code wrapped in `<Code>…</Code>`.\n\n"
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
            "   If `use_kt` is true, also import `keras_tuner as kt` and any additional tuner helpers.\n"
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
            "        X = df.drop(columns=target_cols + [id_col], errors='ignore')\n"
            "        ```\n"
            "    3. Proceed without ever renaming or lowercasing — all subsequent code should refer to columns exactly as in the original files.\n"
            "   - If there is more than one file, decide which ones are for training and testing based on `training_files` and `all_files` and `dataset_metadata` else split the single dataset 80/20 stratified on target.\n"
            "   - Detect & preserve any ID column.\n"
            "Preprocessing MUST ALSO include:\n"
            "  • Label encoding of the target column to integers using\n"
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
            "       • Numeric: `SimpleImputer(strategy='median', add_indicator=True)` → `StandardScaler()`\n"
            "       • Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`\n"
            "   - Fit-transform train and transform val/test.\n"
            "7. **Determine feature dimension**: `input_shape = X_train_proc.shape[1]`.\n"
            "8. **Model Definition**: build an ANN in Keras/TensorFlow (or PyTorch) with at least two hidden layers, including `Dropout` or `BatchNormalization`.\n"
            "9. **Compilation**: `Adam` optimizer, `binary_crossentropy` (or `mse`), metrics `['accuracy']` (or `['RootMeanSquaredError']`).\n"
            "10. **Callbacks & Training**: `EarlyStopping(monitor='val_loss', patience=5)` + `ModelCheckpoint(save_best_only=True)`, up to 100 epochs, record training duration.\n"
            "11. **Evaluation & Logging**: load best weights, extract `training_accuracy`, `training_loss`, `validation_accuracy`, `validation_loss`, save to `results.json`.\n"
            "12. **Prediction & Submission**: transform test set, predict, threshold at 0.5 if classification, write `submission_result.csv` with preserved IDs .\n"
            "      ```python\n"
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
            "         ```   \n"
            "13. **(optional)** If `use_kt` is true, include a complete Keras-Tuner snippet: `HyperModel` subclass, `Hyperband` (or `BayesianOptimization`) setup, `tuner.search()`, and final retraining.\n"
            "No other imports, prose, markdown or keys—just the JSON with a single `notebook_code` field.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "competition_slug": {
                    "type": "string",
                    "description": "The Kaggle competition slug."
                },
                "competition_problem_description": {
                    "type": "string",
                    "description": "Dense competition description giving the core goal."
                },
                "competition_type": {
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
                "use_kt": {
                    "type": "boolean",
                    "description": "Whether to include the Keras-Tuner snippet."
                },
                "notebook_code": {
                    "type": "string",
                    "description": "***The complete runnable Python code wrapped in <Code>…</Code>."
                }
            },
            "required": [
                "competition_slug",
                "competition_problem_description",
                "competition_problem_type",      
                "competition_problem_subtype",
                "dataset_metadata",
                "data_profiles",
                "files_preprocessing_instructions",
                "target_columns",
                "training_files",
                "all_files",
                "examples",
                "use_kt",
                "notebook_code"
            ]
        }
    }


# llm_coding ---> structure_and_label_competition
structure_and_label_competition_schema = {
    "name": "structure_and_label_competition_schema",
    "description": (
        "Given raw Kaggle competition metadata, dataset metadata and a list of files, "
        "return exactly the following fields as JSON:\n"
        "  - competition_type (\"regression\" or \"classification\")\n"
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
            "competition_type": {
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
            "competition_type",
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
