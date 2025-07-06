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
        "  - evaluation_metrics: metrics used to evaluate the solution\n"
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
            "evaluation_metrics": {
                "type": "string",
                "description": "Pick one of the following based on the competition metadata:\n",
                "enum": [
                    "MAP@N – Mean Average Precision",
                    "RMSLE – Root Mean Squared Logarithmic Error",
                    "RMSE – Root Mean Squared Error",
                    "ROC Curve",
                    "MAPE – Mean Absolute Percentage Error",
                    "Accuracy",
                    "MCC – Matthews Correlation Coefficient",
                    "R2 – Coefficient of Determination",
                    "Log Loss",
                    "MedAE – Median Absolute Error",
                    "Micro-averaged F1-Score",
                    "SMAPE – Symmetric Mean Absolute Percentage Error",
                    "MAE – Mean Absolute Error",
                    "Quadratic Weighted Kappa",
                    "Adjusted Rand Index",
                    "AUCROC",
                    "Multi-class Log Loss",
                    "Macro F1 Score",
                    "F1 Score",
                    "Multi-class classification accuracy",
                    "Categorization accuracy",
                    "Classification accuracy"
                ]
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
        "required": ["target_column", "evaluation_metrics", "training_files","submission_file"]
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
                "enum": [
                    "binary-classification",
                    "multiclass-classification",
                    "multi-label-classification",
                    "time-series-forecasting",
                    "continuous-regression",
                    "quantile-regression",
                    "multi-output-regression",
                    "ordinal-regression",
                    "missing-value-imputation"
                ],
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
        "  - competition_problem_subtype (lower-case, single, concise, lowercase‐and‐hyphenated phrase (e.g. “binary classification”, “multiclass classification”, “multi-label classification”, “time-series forecasting”, “continuous regression”, “ordinal regression”, etc. or any other)\n"
        "  - competition_problem_description (dense, non-repetitive description of the goal)\n"
        "  - evaluation_metrics metrics used to evaluate the solution\n"
        "  - dataset_metadata (plain-English paragraph rewrite of the original)\n"
        "  - competition_dataset_type (one of: Tabular, Time-series, Text, Image, Audio, Video, Geospatial, Graph, Multimodal)\n"
        "  - target_column (array of the exact label column name(s) in the training files)\n"
        "  - files_list (the raw file names discovered on the data tab)\n"
        "  - all_files - All files used for the competition available for download, may not include all of them\n"
        "  - training_files (subset of files_list to load as training tables)\n"
        "  - submission_files - submission guide/example file `\n"                
        "  - files_preprocessing_instructions (plain-English instructions to prep those files)\n"
        "No extra keys, no prose—just that JSON object."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "competition_problem_type": {
                "type": "string",
                "enum": ["regression", "classification"],
                "description": "Either 'regression' or 'classification'."
            },
            "competition_problem_subtype": {
                "type": "string",
                "description": "Pay attention to the problem evaluation and submission wording to pick the exact subtype. Be careful not to confuse:\n"
                            "- multiclass vs. multi-label classification\n"
                            "- continuous-regression vs. quantile-regression\n"
                            "- multi-output-regression vs. multi-label-classification\n"
                            "- ordinal-regression vs. multiclass-classification\n"
                            "- missing-value-imputation vs. regression\n"
                            "- time-series-forecasting vs. non-temporal regression",
                "oneOf": [
                    {
                    "const": "binary-classification",
                    "description": "Predict one of two mutually exclusive classes for each example."
                    },
                    {
                    "const": "multiclass-classification",
                    "description": "Predict one class out of more than two mutually exclusive classes."
                    },
                    {
                    "const": "multi-label-classification",
                    "description": "Assign one or more non-exclusive labels to each example."
                    },
                    {
                    "const": "time-series-forecasting",
                    "description": "Predict future values given observations ordered in time."
                    },
                    {
                    "const": "continuous-regression",
                    "description": "Predict a single continuous numeric target."
                    },
                    {
                    "const": "quantile-regression",
                    "description": "Predict specified quantiles (e.g. 0.1, 0.5, 0.9) of a continuous distribution."
                    },
                    {
                    "const": "multi-output-regression",
                    "description": "Predict multiple continuous targets simultaneously."
                    },
                    {
                    "const": "ordinal-regression",
                    "description": "Predict discrete ordered categories (e.g. ratings)."
                    },
                    {
                    "const": "missing-value-imputation",
                    "description": "Predict and fill in missing entries in the dataset."
                    }
                ]
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
            "evaluation_metrics": {
                "type": "string",
                "description": "Pick one of the following based on the competition metadata:\n",
                "enum": [
                    "MAP@N – Mean Average Precision",
                    "RMSLE – Root Mean Squared Logarithmic Error",
                    "RMSE – Root Mean Squared Error",
                    "ROC Curve",
                    "MAPE – Mean Absolute Percentage Error",
                    "Accuracy",
                    "MCC – Matthews Correlation Coefficient",
                    "R2 – Coefficient of Determination",
                    "Log Loss",
                    "MedAE – Median Absolute Error",
                    "Micro-averaged F1-Score",
                    "SMAPE – Symmetric Mean Absolute Percentage Error",
                    "MAE – Mean Absolute Error",
                    "Quadratic Weighted Kappa",
                    "Adjusted Rand Index",
                    "AUCROC",
                    "Multi-class Log Loss",
                    "Macro F1 Score",
                    "F1 Score",
                    "Multi-class classification accuracy",
                    "Categorization accuracy",
                    "Classification accuracy"
                ]
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
                "description": "**Based on the files_list, all_files, and dataset_metadata, give  an array of exact names of all training tabular files that need to be downloaded, ensure that the listed files in the dataset_metadata correspond to the ones in files_list, if not go with the files most similar in the files_list"
            },
            "submission_file": {
                "type": "string",
                "description": "**Based on the files_list, all_files, and dataset_metadata, give  an exact name of the submission example file that needs to be downloaded, ensure that the listed file in the dataset_metadata correspond to the one in files_list, if not go with the file most similar in the files_list"
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
            "evaluation_metrics",
            "dataset_metadata",
            "competition_dataset_type",
            "target_column",
            "files_list",
            "all_files",
            "training_files",
            "submission_file", 
            "files_preprocessing_instructions"
            
        ]
    }
}

tools = [
    {
        "name": "generate_keras_schema",  
        "type": "function",
        "description": (
            "***Generate and save a runnable deep learning model using Keras in Python code wrapped in <Code>…</Code> in a single `notebook_code` JSON field:\\n"  
            "The generated code must implement:\n"
            "1. **Reproducibility**: set seeds for Python, NumPy, scikit-learn, and TensorFlow (or PyTorch).\n"
            "2. **Imports**:\\n\"  \n"
            "   - `pandas`, `numpy`\\n\"  \n"
            "   - `sklearn.model_selection.train_test_split`\\n\"  \n"
            "   - `sklearn.impute.SimpleImputer`\\n\"  \n"
            "   - `sklearn.compose.ColumnTransformer`\\n\"  \n"
            "   - `sklearn.preprocessing.StandardScaler`,`OneHotEncoder`,`LabelEncoder`  ← **added here**\\n\"  \n"
            "   - `sklearn.pipeline.Pipeline`\\n\"  \n"
            "   - `tensorflow` (or `torch`)\\n\"  \n"
            "   - `tensorflow.keras.callbacks.EarlyStopping,ModelCheckpoint`\\n\"  \n"
            "   - `json`, `time`\\n\"  \n"
            "   When using OneHotEncoding, use sparse_output=False instead of sparse\n"
           "3. Data Loading, Split & Target Encoding:\n"
            "Read each file in files_list into train_dfs\n"
            "If any filename endswith 'test.csv', load it into df_test, else df_test=None\n"
            "Infer id_col & target_columns from submission_example header\n"
            "df = pd.concat(train_dfs, ignore_index=True)\n"
            "# Target encoding immediately after df is final:\n"
            "col = target_columns[0]\n"
            "if competition_problem_subtype in ['binary-classification']:\n"
            "    from sklearn.preprocessing import LabelEncoder\n"
            "    le=LabelEncoder().fit(df[col].astype(str))\n"
            "    y_enc=le.transform(df[col].astype(str)).astype(int)\n"
            "    classes_=le.classes_\n"
            "elif competition_problem_subtype in ['multiclass-classification','multiclass classification','ordinal-regression']:\n"
            "    from sklearn.preprocessing import LabelEncoder\n"
            "    le=LabelEncoder().fit(df[col].astype(str))\n"
            "    y_enc=le.transform(df[col].astype(str))\n"
            "    classes_=le.classes_\n"
            "elif competition_problem_subtype in ['multi-label classification']:\n"
            "    from sklearn.preprocessing import MultiLabelBinarizer\n"
            "    mlb=MultiLabelBinarizer()\n"
            "    y_enc=mlb.fit_transform(df[target_columns])\n"
            "    classes_=mlb.classes_\n"
            "elif competition_problem_subtype in ['continuous-regression','quantile-regression','multi-output regression','missing-value-imputation']:\n"
            "    y_values = df[target_columns].astype(float).values\n"
            "    y_enc = np.log1p(y_values) if np.all(y_values >= 0) else y_values\n"
            "elif competition_problem_subtype in ['time-series-forecasting','multivariate-time-series-forecasting']:\n"
            "    y_enc=df[target_columns].values\n"
            "else:\n"
            "    y_enc=df[target_columns].values\n"
            "X=df.drop(columns=target_columns+[id_col],errors='ignore')\n"
            "# now either use provided df_test or split off 20% for test:\n"
            "if df_test is None:\n"
            "    X_train,X_val,y_train,y_val=train_test_split(\n"
            "        X,y_enc,\n"
            "        test_size=0.2,\n"
            "        stratify=y_enc if competition_problem_subtype in ['binary-classification','multiclass-classification','multiclass classification'] else None,\n"
            "        random_state=42)\n"
            "    train_ids=X_train[id_col]\n"
            "    test_ids =X_val[id_col]\n"
            "else:\n"
            "    X_train=X\n"
            "    y_train=y_enc\n"
            "    train_ids=df[id_col]\n"
            "    test_ids =df_test[id_col]\n"
            "    X_val   =df_test.drop(columns=target_columns+[id_col],errors='ignore')\n"
            "    y_val   = None  # explicitly set\n"
            "\n"
            "4. Feature Engineering:\n"
                "Automatically drop columns with all missing values\n"
                "Identify categorical columns and remove those with extremely high cardinality (eg >50 unique)\n"
                "Optionally apply any additional simple transformations you deem useful\n"
            "5. **Preprocessing Pipeline**:\n"
                "   - Auto-detect numeric vs. categorical via `df.select_dtypes`.\n"
                "   - Build a `ColumnTransformer` with median‐imputed & scaled numerics, and most‐frequent‐imputed & OHE categoricals (cap at 50 cats).\n"
                "   - Fit on train → transform train/val/test.\n"
            "6. **Fix numbering**: ensure your sections run 0→11 with no gaps or duplicates.\n"
            "7. **Model Architecture:**\n"
                "- Build at least two hidden layers with BatchNormalization and Dropout after each\n"
                "- Set output units = number of target_columns for multilabel/multiclass, else 1\n"
                "- Choose depth & width by data shape: shallow/narrow for small datasets, deeper/wider for large datasets, scale units ≈ min(features×2,1024)\n"
                "- Leverage provided `examples` but adjust architecture based on dataset size, feature count, and target count\n"
                "- **Architectural Guidelines:**\n"
                "   - **Choose by data size:**\n"
                "     • If `n_samples < 10000` or `n_features < 100`:\n"
                "         – Build **two** Dense layers of sizes:\n"
                "             [min(n_features*2, 128), min(n_features, 64)]\n"
                "         – **No** BatchNormalization; Dropout ≤ 0.3\n"
                "     • Else:\n"
                "         – Build **2–4** Dense layers of sizes:\n"
                "             [min(n_features*i, 1024) for i in (2, 1, 0.5, 0.25)] (drop any <16 units)\n"
                "         – After each: BatchNormalization() + Dropout(≤0.4)\n"
                "\n"
                "***For all hidden layers (except the final output), use ReLU activation***\n"
                "  - **Task subtype → head, loss, batch & metrics:**\n"
                "    **(Note: activation applies only to the final/output layer)**\n"
                "    * **binary classification:**\n"
                "        – activation=sigmoid, loss=binary_crossentropy\n"
                "        – batch_size=64–256, metrics=['accuracy', tf.keras.metrics.AUC(), tfa.metrics.MatthewsCorrelationCoefficient()]\n"
                "    * **multiclass classification (MAP@N):**\n"
                "        – activation=softmax, loss=sparse_categorical_crossentropy\n"
                "        – batch_size=32–128\n"
                "        – dynamically compute top_k as: \n"
                "        – num_classes = len(np.unique(y_enc)) if isinstance(y_enc, (list, np.ndarray)) else 3\n"
                "        – top_k = min(num_classes, 5)\n"
                "        – metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f'top_{top_k}_accuracy')]\n"
                "        – at inference: take the top-`top_k` softmax probabilities for submission\n"
                "    * **multilabel classification:**\n"
                "        – activation=sigmoid, loss=binary_crossentropy\n"
                "        – batch_size=64–256, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_classes)]\n"
                "    * **regression:**\n"
                "        – activation=linear, loss=mean_squared_error\n"
                "        – batch_size=32–256, metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]\n"
                "    * **time-series forecasting:**\n"
                "        – use chronological split\n"
                "        – activation=linear, loss=mean_squared_error\n"
                "        – epochs=10–50, metrics=[tf.keras.metrics.RootMeanSquaredError()]\n"
                ")"
            "8. **Compile the model with the Adam optimizer and the chosen loss and metrics\n"
            "9. **Callbacks & Training**:\\n"
            "   start_time = time.time()  # capture before fit\\n"
            "   if y_val is not None:\\n"
            "       history = model.fit(X_train_proc, y_train, validation_data=(X_val_proc, y_val), epochs=100, callbacks=callbacks, verbose=2)\\n"
            "   else:\\n"
            "       history = model.fit(X_train_proc, y_train, validation_split=0.2, epochs=100, callbacks=callbacks, verbose=2)\\n"
            "   duration = time.time() - start_time  # compute after fit\\n"
            "10. **Evaluation & Logging**:\\n\"  \n"
            "   Don't user tensorflow_addons"
            "   Turn on the verbose and save the training and validtion accuracy and log of the last epoch in a json file (results.json). It will have the following keys: {training_accuracy, training_loss,validation_accuracy and validation_loss}\n"
            "   with open('results.json','w') as f: json.dump(results,f)\\n\"  \n"
            "11. **Prediction & Submission**:\n"
                "raw_preds = model.predict(X_test_proc)\n"
                "***if competition_problem_subtype == ['multiclass','multiclass classification','multi-label classification` ]: final = le.inverse_transform(raw_preds.argmax(axis=1))\n"
                "***elif competition_problem_subtype == 'binary-classification': final = (raw_preds.flatten() > 0.5).astype(int)\n"
                "***else: final = raw_preds.flatten()\n"
                "if competition_problem_subtype in ['continuous-regression','quantile-regression','multi-output regression','missing-value-imputation']:\n"
                "    final = np.expm1(np.clip(final, a_min=None, a_max=20))  # inverse transform\n"
                "if len(target_columns) == 1:\n"
                "    submission = pd.DataFrame({id_col: test_ids, target_columns[0]: final})\n"
                "else:\n"
                "    submission = pd.DataFrame(final, columns=target_columns)\n"
                "    submission.insert(0, id_col, test_ids)\n"
                "submission.to_csv('submission_result.csv', index=False)\n"
            
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "competition_problem_description": {
                    "type": "string",
                    "description": "Dense competition description giving the core goal."
                },
                "competition_problem_subtype": {
                    "type": "string",
                    "description": "One of:\n"
                        "  - binary-classification\n"
                        "  - multiclass-classification\n"
                        "  - multiclass classification\n"
                        "  - multi-label classification\n"
                        "  - continuous-regression\n"
                        "  - quantile-regression\n"
                        "  - multi-output regression\n"
                        "  - time-series-forecasting\n"
                        "  - multivariate-time-series-forecasting\n"
                        "  - ordinal-regression\n"
                        "  - missing-value-imputation\n"
                        "Rely on this to choose splits, loss, activation, etc."
                },
                "dataset_metadata": {
                    "type": "string",
                    "description": "Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided"
                },
                "data_profiles": {
                    "type": "object",
                    "additionalProperties": False, 
                    "properties": {}  
                },
                "files_preprocessing_instructions": {
                    "type": "string",
                    "description": "Instructions for how to preprocess the raw files."
                },
                "submission_example": {
                    "type": "string", 
                    "description": (  
                        "Contains the target columns ***not including the id column*** that need to be predicted and the example format of columns and values that needs to be outputted to the submission_results.csv`\n\
                        Rely on the `submission_example` for how to format the sumbission and pay attentiton for what types of values there are\n"                
                    )
                },
                "files_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": " list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset"
                },
                "examples": {
                    "type": "array",
                    "description": "Retrieved preprocessing and code snippets from solutions of top similar competitions, rely on them ",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "score":                {"type":"number"},
                            "preprocessing_steps":  {
                                "type":"array",
                                "items":{"type":"string"}
                            },
                            "model_layers_code":    {"type":"string"}
                        },
                        "required":["score","preprocessing_steps","model_layers_code"]
                    }
                },
                "notebook_code": {
                    "type": "string",
                    "description": "The complete runnable Python notebook code wrapped in <Code>…</Code>."
                }
            },
            "required": [
                "competition_problem_description",
                "competition_problem_subtype",
                "dataset_metadata",
                "data_profiles",
                "files_preprocessing_instructions",
                "submission_example",
                "files_list",
                "examples",
                "notebook_code"
            ]
        },
        "strict": True     # ← enforce valid JSON
    }
] 



# tools = [
#     {
#         "name": "generate_keras_schema",  
#         "type": "function",
#         "description": (
#             "***Generate and save a runnable Python code wrapped in <Code>…</Code> in a singe `notebook_code` json field:\\n"  
#             "The generated code must implement:\n"
#             "1. **Reproducibility**: set seeds for Python, NumPy, scikit-learn, and TensorFlow (or PyTorch).\n"
#             "2. **Imports**:\\n\"  \n"
#             "   - `pandas`, `numpy`\\n\"  \n"
#             "   - `sklearn.model_selection.train_test_split`\\n\"  \n"
#             "   - `sklearn.impute.SimpleImputer`\\n\"  \n"
#             "   - `sklearn.compose.ColumnTransformer`\\n\"  \n"
#             "   - `sklearn.preprocessing.StandardScaler`,`OneHotEncoder`,`LabelEncoder`  ← **added here**\\n\"  \n"
#             "   - `sklearn.pipeline.Pipeline`\\n\"  \n"
#             "   - `tensorflow` (or `torch`)\\n\"  \n"
#             "   - `tensorflow.keras.callbacks.EarlyStopping,ModelCheckpoint`\\n\"  \n"
#             "   - `json`, `time`\\n\"  \n"
#             "   When using OneHotEncoding, use sparse_output=False instead of sparse\n"
#            "3. Data Loading, Split & Target Encoding:\n"
#             "Read each file in files_list into train_dfs\n"
#             "If any filename endswith 'test.csv', load it into df_test, else df_test=None\n"
#             "Infer id_col & target_columns from submission_example header\n"
#             "df = pd.concat(train_dfs, ignore_index=True)\n"
#             "# Target encoding immediately after df is final:\n"
#             "col = target_columns[0]\n"
#             "if competition_problem_subtype=='binary-classification':\n"
#             "    from sklearn.preprocessing import LabelEncoder\n"
#             "    le=LabelEncoder().fit(df[col].astype(str))\n"
#             "    y_enc=le.transform(df[col].astype(str)).astype(int)\n"
#             "    classes_=le.classes_\n"
#             "elif competition_problem_subtype in ['multiclass-classification','multiclass classification','ordinal-regression']:\n"
#             "    from sklearn.preprocessing import LabelEncoder\n"
#             "    le=LabelEncoder().fit(df[col].astype(str))\n"
#             "    y_enc=le.transform(df[col].astype(str))\n"
#             "    classes_=le.classes_\n"
#             "elif competition_problem_subtype=='multi-label classification':\n"
#             "    from sklearn.preprocessing import MultiLabelBinarizer\n"
#             "    mlb=MultiLabelBinarizer()\n"
#             "    y_enc=mlb.fit_transform(df[target_columns])\n"
#             "    classes_=mlb.classes_\n"
#             "elif competition_problem_subtype in ['continuous-regression','quantile-regression','multi-output regression','missing-value-imputation']:\n"
#             "    y_enc=df[target_columns].astype(float).values\n"
#             "elif competition_problem_subtype in ['time-series-forecasting','multivariate-time-series-forecasting']:\n"
#             "    y_enc=df[target_columns].values\n"
#             "else:\n"
#             "    y_enc=df[target_columns].values\n"
#             "X=df.drop(columns=target_columns+[id_col],errors='ignore')\n"
#             "# now either use provided df_test or split off 20% for test:\n"
#             "if df_test is None:\n"
#             "    X_train,X_val,y_train,y_val=train_test_split(\n"
#             "        X,y_enc,\n"
#             "        test_size=0.2,\n"
#             "        stratify=y_enc if competition_problem_subtype in ['binary-classification','multiclass-classification','multiclass classification'] else None,\n"
#             "        random_state=42)\n"
#             "    train_ids=X_train[id_col]\n"
#             "    test_ids =X_val[id_col]\n"
#             "else:\n"
#             "    X_train=X\n"
#             "    y_train=y_enc\n"
#             "    train_ids=df[id_col]\n"
#             "    test_ids =df_test[id_col]\n"
#             "    X_val   =df_test.drop(columns=target_columns+[id_col],errors='ignore')\n"
#             "\n"
#             "4. Feature Engineering:\n"
#                 "Automatically drop columns with all missing values\n"
#                 "Identify categorical columns and remove those with extremely high cardinality (eg >50 unique)\n"
#                 "Optionally apply any additional simple transformations you deem useful\n"
#             "5. **Preprocessing Pipeline**:\n"
#                 "   - Auto-detect numeric vs. categorical via `df.select_dtypes`.\n"
#                 "   - Build a `ColumnTransformer` with median‐imputed & scaled numerics, and most‐frequent‐imputed & OHE categoricals (cap at 50 cats).\n"
#                 "   - Fit on train → transform train/val/test.\n"
#             "6. **Fix numbering**: ensure your sections run 0→11 with no gaps or duplicates.\n"
#             "7. **Model Architecture:**\n"
#                 "- Build at least two hidden layers with BatchNormalization and Dropout after each\n"
#                 "- Set output units = number of target_columns for multilabel/multiclass, else 1\n"
#                 "- Choose depth & width by data shape: shallow/narrow for small datasets, deeper/wider for large datasets, scale units ≈ min(features×2,1024)\n"
#                 "- Leverage provided `examples` but adjust architecture based on dataset size, feature count, and target count\n"
#                 "- **Architectural Guidelines:**\n"
#                 "   - **Choose by data size:**\n"
#                 "     • If `n_samples < 10000` or `n_features < 100`:\n"
#                 "         – Build **two** Dense layers of sizes:\n"
#                 "             [min(n_features*2, 128), min(n_features, 64)]\n"
#                 "         – **No** BatchNormalization; Dropout ≤ 0.3\n"
#                 "     • Else:\n"
#                 "         – Build **2–4** Dense layers of sizes:\n"
#                 "             [min(n_features*i, 1024) for i in (2, 1, 0.5, 0.25)] (drop any <16 units)\n"
#                 "         – After each: BatchNormalization() + Dropout(≤0.4)\n"
#                 "\n"
#                 "***For all hidden layers (except the final output), use ReLU activation***\n"
#                 "  - **Task subtype → head, loss, batch & metrics:**\n"
#                 "    **(Note: activation applies only to the final/output layer)**\n"
#                 "    * **binary classification:**\n"
#                 "        – activation=sigmoid, loss=binary_crossentropy\n"
#                 "        – batch_size=64–256, metrics=['accuracy', tf.keras.metrics.AUC(), tfa.metrics.MatthewsCorrelationCoefficient()]\n"
#                 "    * **multiclass classification (MAP@N):**\n"
#                 "        – activation=softmax, loss=sparse_categorical_crossentropy\n"
#                 "        – batch_size=32–128, metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=N, name=f'top_{N}_accuracy')]\n"
#                 "        – at inference: take the top-N softmax probabilities for submission\n"
#                 "    * **multilabel classification:**\n"
#                 "        – activation=sigmoid, loss=binary_crossentropy\n"
#                 "        – batch_size=64–256, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_classes)]\n"
#                 "    * **regression:**\n"
#                 "        – activation=linear, loss=mean_squared_error\n"
#                 "        – batch_size=32–256, metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]\n"
#                 "    * **time-series forecasting:**\n"
#                 "        – use chronological split\n"
#                 "        – activation=linear, loss=mean_squared_error\n"
#                 "        – epochs=10–50, metrics=[tf.keras.metrics.RootMeanSquaredError()]\n"
#                 ")"
#             "8. **Compile the model with the Adam optimizer and the chosen loss and metrics\n"
#             "9. **Callbacks & Training**:\\n\"  \n"
#             "   start_time = time.time()  ← **capture before fit**\\n\"  \n"
#             "   history = model.fit(X_train_proc, y_train, validation_data=(X_val_proc, y_val), epochs=100, callbacks=callbacks)\\n\"  \n"
#             "   duration = time.time() - start_time  ← **compute after fit**\\n\"  \n"
#             "10. **Evaluation & Logging**:\\n\"  \n"
#             "   Don't user tensorflow_addons"
#             "   Turn on the verbose and save the training and validtion accuracy and log of the last epoch in a json file (results.json). It will have the following keys: {training_accuracy, training_loss,validation_accuracy and validation_loss}\n"
#             "   with open('results.json','w') as f: json.dump(results,f)\\n\"  \n"
#             "11. **Prediction & Submission**:\n"
#                 "raw_preds = model.predict(X_test_proc)\n"
#                 "***if competition_problem_subtype == ['multiclass','multiclass classification','multi-label classification` ]: final = le.inverse_transform(raw_preds.argmax(axis=1))\n"
#                 "***elif competition_problem_subtype == 'binary-classification': final = (raw_preds.flatten() > 0.5).astype(int)\n"
#                 "***else: final = raw_preds.flatten()\n"
#                 "if len(target_columns) == 1:\n"
#                 "    submission = pd.DataFrame({id_col: test_ids, target_columns[0]: final})\n"
#                 "else:\n"
#                 "    submission = pd.DataFrame(final, columns=target_columns)\n"
#                 "    submission.insert(0, id_col, test_ids)\n"
#                 "submission.to_csv('submission_result.csv', index=False)\n"
            
#         ),
#         "parameters": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "competition_problem_description": {
#                     "type": "string",
#                     "description": "Dense competition description giving the core goal."
#                 },
#                 "competition_problem_subtype": {
#                     "type": "string",
#                     "description": "One of:\n"
#                         "  - binary-classification\n"
#                         "  - multiclass-classification\n"
#                         "  - multiclass classification\n"
#                         "  - multi-label classification\n"
#                         "  - continuous-regression\n"
#                         "  - quantile-regression\n"
#                         "  - multi-output regression\n"
#                         "  - time-series-forecasting\n"
#                         "  - multivariate-time-series-forecasting\n"
#                         "  - ordinal-regression\n"
#                         "  - missing-value-imputation\n"
#                         "Rely on this to choose splits, loss, activation, etc."
#                 },
#                 "dataset_metadata": {
#                     "type": "string",
#                     "description": "Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided"
#                 },
#                 "data_profiles": {
#                     "type": "object",
#                     "additionalProperties": False, 
#                     "properties": {}  
#                 },
#                 "files_preprocessing_instructions": {
#                     "type": "string",
#                     "description": "Instructions for how to preprocess the raw files."
#                 },
#                 "submission_example": {
#                     "type": "string", 
#                     "description": (  
#                         "Contains the target columns ***not including the id column*** that need to be predicted and the example format of columns and values that needs to be outputted to the submission_results.csv`\n\
#                         Rely on the `submission_example` for how to format the sumbission and pay attentiton for what types of values there are\n"                
#                     )
#                 },
#                 "files_list": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": " list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset"
#                 },
#                 "examples": {
#                     "type": "array",
#                     "description": "Retrieved preprocessing and code snippets from solutions of top similar competitions, rely on them ",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "score":                {"type":"number"},
#                             "preprocessing_steps":  {
#                                 "type":"array",
#                                 "items":{"type":"string"}
#                             },
#                             "model_layers_code":    {"type":"string"}
#                         },
#                         "required":["score","preprocessing_steps","model_layers_code"]
#                     }
#                 },
#                 "notebook_code": {
#                     "type": "string",
#                     "description": "The complete runnable Python notebook code wrapped in <Code>…</Code>."
#                 }
#             },
#             "required": [
#                 "competition_problem_description",
#                 "competition_problem_subtype",
#                 "dataset_metadata",
#                 "data_profiles",
#                 "files_preprocessing_instructions",
#                 "submission_example",
#                 "files_list",
#                 "examples",
#                 "notebook_code"
#             ]
#         },
#         "strict": True     # ← enforce valid JSON
#     }
# ] 


# # llm_coding ----> solve_competition_keras
# generate_keras_schema = {
#         "name": "generate_keras_schema",   
#         "description": (
#             "***Generate and save a runnable Python code wrapped in <Code>…</Code> in a singe `notebook_code` json field:\\n"  
#             "The generated code must implement:\n"
#             "1. **Reproducibility**: set seeds for Python, NumPy, scikit-learn, and TensorFlow (or PyTorch).\n"
#             "2. **Imports**:\\n\"  \n"
#             "   - `pandas`, `numpy`\\n\"  \n"
#             "   - `sklearn.model_selection.train_test_split`\\n\"  \n"
#             "   - `sklearn.impute.SimpleImputer`\\n\"  \n"
#             "   - `sklearn.compose.ColumnTransformer`\\n\"  \n"
#             "   - `sklearn.preprocessing.StandardScaler`,`OneHotEncoder`,`LabelEncoder`  ← **added here**\\n\"  \n"
#             "   - `sklearn.pipeline.Pipeline`\\n\"  \n"
#             "   - `tensorflow` (or `torch`)\\n\"  \n"
#             "   - `tensorflow.keras.callbacks.EarlyStopping,ModelCheckpoint`\\n\"  \n"
#             "   - `json`, `time`\\n\"  \n"
#             "3. Data Loading, Split & Target Encoding:\n"
#                 "Read each file in files_list into train_dfs\n"
#                 "If any filename endswith 'test.csv', load it into df_test, else df_test=None\n"
#                 "Infer id_col & target_columns from submission_example header\n"
#                 "df = pd.concat(train_dfs, ignore_index=True)\n"
#                 "# Now df holds only training data\n"
#                 "# Target encoding immediately after df is final:\n"
#                 "col = target_columns[0]\n"
#                 "if competition_problem_subtype in ['time-series-forecasting','multivariate-time-series-forecasting']:\n"
#                 "    # chronological split\n"
#                 "    cutoff = int(len(df) * 0.8)\n"
#                 "    X_train,   y_train = X[:cutoff],   y_enc[:cutoff]\n"
#                 "    X_val,     y_val   = X[cutoff:],   y_enc[cutoff:]\n"
#                 "else:\n"
#                 "    # random split (stratify only for binary/multiclass)\n"
#                 "    kwargs = dict(test_size=0.2, random_state=42)\n"
#                 "    if competition_problem_subtype in ['binary-classification','multiclass']:\n"
#                 "        kwargs['stratify'] = y_enc\n"
#                 "    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, **kwargs)\n"
#                 "# Use provided df_test or split df for test set:\n"
#                 "if df_test is None:\n"
#                 "    X = df.drop(columns=target_columns + [id_col], errors='ignore')\n"
#                 "    X_train, X_val, y_train, y_val = train_test_split(\n"
#                 "        X, y_enc,\n"
#                 "        test_size=0.2,\n"
#                 "        stratify=y_enc if competition_problem_subtype in ['binary-classification','multiclass'] else None,\n"
#                 "        random_state=42)\n"
#                 "    train_ids = X_train[id_col]\n"
#                 "    test_ids  = X_val[id_col]\n"
#                 "else:\n"
#                 "    X_train = df.drop(columns=target_columns + [id_col], errors='ignore')\n"
#                 "    y_train = y_enc\n"
#                 "    train_ids = df[id_col]\n"
#                 "    test_ids  = df_test[id_col]\n"
#                 "    X_test   = df_test.drop(columns=target_columns + [id_col], errors='ignore')\n"
#             "4. Feature Engineering:\n"
#                 "Automatically drop columns with all missing values\n"
#                 "Identify categorical columns and remove those with extremely high cardinality (eg >50 unique)\n"
#                 "Optionally apply any additional simple transformations you deem useful\n"
#             "5. **Preprocessing Pipeline**:\n"
#                 "   - Auto-detect numeric vs. categorical via `df.select_dtypes`.\n"
#                 "   - Build a `ColumnTransformer` with median‐imputed & scaled numerics, and most‐frequent‐imputed & OHE categoricals (cap at 50 cats).\n"
#                 "   - Fit on train → transform train/val/test.\n"
#             "6. **Fix numbering**: ensure your sections run 0→11 with no gaps or duplicates.\n"
#             "7. **Model Activation Functions & Sizing:**\n"
#                 "- Build at least two hidden layers with BatchNormalization and Dropout after each\n"
#                 "- Set output units = number of target_columns for multilabel/multiclass, else 1\n"
#                 "For the last model layer pick one of these activation functions based on the `competition_problem_subtype`\n"
#                 "- ***Use softmax activation for ['multiclass','multiclass classification','multi-label classification` ] classification\n"
#                 "- ***Use sigmoid activation for binary classification or multilabel\n"
#                 "- ***Use linear activation for regression or other continuous targets\n"
#                 "- ****Choose depth and width by data shape: shallow/narrow for small datasets, deeper/wider for large datasets, scale units ≈ min(features×2, 1024)\n"
#                 "- Leverage provided `examples` preprocessing steps and model code but adjust architecture based on dataset size, feature count, and target count\n"
#             "8. Loss Functions and Compilation:\n"
#                 "- For regression: set loss to 'mse' and metrics to ['RootMeanSquaredError']\n"
#                 "- For binary classification: set loss to 'binary_crossentropy' and metrics to ['accuracy']\n"
#                 "- For multiclass classification: set loss to 'sparse_categorical_crossentropy' and metrics to ['accuracy']\n"
#                 "- For multilabel classification: set loss to 'binary_crossentropy' and metrics to ['accuracy']\n"
#                 "- Compile the model with the Adam optimizer and the chosen loss and metrics\n"
#             "9. **Callbacks & Training**:\\n\"  \n"
#             "   start_time = time.time()  ← **capture before fit**\\n\"  \n"
#             "   history = model.fit(X_train_proc, y_train, validation_data=(X_val_proc, y_val), epochs=100, callbacks=callbacks)\\n\"  \n"
#             "   duration = time.time() - start_time  ← **compute after fit**\\n\"  \n"
#             "10. **Evaluation & Logging**:\\n\"  \n"
#             "   results = {\\n\"  \n"
#             "       'training_accuracy': train_accuracy,\\n\"  \n"
#             "       'validation_accuracy': val_accuracy,\\n\"  \n"
#             "       'validation_loss': val_loss,\\n\"  \n"
#             "       'training_loss': train_loss,\\n\"  \n"
#             "       'training_duration': duration\\n\"  \n"
#             "   }\\n\"  \n"
#             "   with open('results.json','w') as f: json.dump(results,f)\\n\"  \n"
#             "11. **Prediction & Submission**:\n"
#                 "raw_preds = model.predict(X_test_proc)\n"
#                 "***if competition_problem_subtype == ['multiclass','multiclass classification','multi-label classification` ]: final = le.inverse_transform(raw_preds.argmax(axis=1))\n"
#                 "***elif competition_problem_subtype == 'binary-classification': final = (raw_preds.flatten() > 0.5).astype(int)\n"
#                 "***else: final = raw_preds.flatten()\n"
#                 "if len(target_columns) == 1:\n"
#                 "    submission = pd.DataFrame({id_col: test_ids, target_columns[0]: final})\n"
#                 "else:\n"
#                 "    submission = pd.DataFrame(final, columns=target_columns)\n"
#                 "    submission.insert(0, id_col, test_ids)\n"
#                 "submission.to_csv('submission_result.csv', index=False)\n"
            
#         ),
#         "parameters": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "competition_problem_description": {
#                     "type": "string",
#                     "description": "Dense competition description giving the core goal."
#                 },
#                 "competition_problem_subtype": {
#                     "type": "string",
#                     "description": "One of:\n"
#                         "  - binary-classification\n"
#                         "  - multiclass-classification\n"
#                         "  - multiclass classification\n"
#                         "  - multi-label classification\n"
#                         "  - continuous-regression\n"
#                         "  - quantile-regression\n"
#                         "  - multi-output regression\n"
#                         "  - time-series-forecasting\n"
#                         "  - multivariate-time-series-forecasting\n"
#                         "  - ordinal-regression\n"
#                         "  - missing-value-imputation\n"
#                         "Rely on this to choose splits, loss, activation, etc."
#                 },
#                 "dataset_metadata": {
#                     "type": "string",
#                     "description": "Full NLP explanation of the dataset, the columns that need to be predicted and the training files provided"
#                 },
#                 "data_profiles": {
#                     "type": "object",
#                     "description": "A mapping filename → dataset schema & target summary of each file provided in the competition "
#                 },
#                 "files_preprocessing_instructions": {
#                     "type": "string",
#                     "description": "Instructions for how to preprocess the raw files."
#                 },
#                 "submission_example": {
#                     "type": "string", 
#                     "description": (  
#                         "Contains the target columns ***not including the id column*** that need to be predicted and the example format of columns and values that needs to be outputted to the submission_results.csv`\n\
#                         Rely on the `submission_example` for how to format the sumbission and pay attentiton for what types of values there are\n"                
#                     )
#                 },
#                 "files_list": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": " list of all files included in the competition, decide whether there are testing files and whether you need to split the training dataset"
#                 },
#                 "examples": {
#                     "type": "array",
#                     "description": "Retrieved preprocessing and code snippets from solutions of top similar competitions, rely on them ",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "kernel_ref":           {"type":"string"},
#                             "score":                {"type":"number"},
#                             "preprocessing_steps":  {
#                                 "type":"array",
#                                 "items":{"type":"string"}
#                             },
#                             "model_layers_code":    {"type":"string"}
#                         },
#                         "required":["kernel_ref","score","preprocessing_steps","model_layers_code"]
#                     }
#                 },

#                 "notebook_code": {
#                     "type": "string",
#                     "description": "***The complete runnable Python code wrapped in <Code>…</Code>, saved in the `notebook_code` json field.***"
#                 }
#             },
#             "required": [
#                 "notebook_code"
#             ]
#         },
#         "strict": True
#     }



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
            "  - Emit ONLY a single JSON object with exactly one field: "
            "  - ***`tuner_code`: a string containing the **full** runnable Python notebook code wrapped in `<Code>…</Code>`\n"
            "  - This must include **all** original data loading, preprocessing, model definition, callbacks, training, **and** the Keras-Tuner integration (imports, HyperModel wrapper, tuner setup, search, and best_model rebuild), as well as final evaluation and `submission.to_csv`.\n"        "    (including imports, HyperModel subclass, tuner setup, search, and final retrain)\n"
            "  - Keep the structure the same as the original Keras code, including the training timing, saving the result into a submission file \n"
        "   0.1. Use `chosen_profile[\"params\"]` to drive every `hp.*` call below.\n" 
        "***IMPORTANT CODING RULES:***\n"
            "  - First, select `hyperparameter_bank` by comparing each bank-entry’s `tags` to `competition_problem_type`, `competition_problem_subtype`, and whether the data is tabular/text.\n"
            "  - Use the model layers, compile, and fit, provided by the `hyperparameter_bank` if they fit better than the original Keras code layers \n" 
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
            "dataset_metadata":                 
            {
                "type": "string"
                "description"
            },
            "data_profiles":                    
            {
                "type": "object",
                "description": "A mapping filename → dataset schema & target summary of each file provided in the competition "
            },
            "existing_solution_code":           
            {
                "type": "string",
                "description": "Contains an existing Keras solution that should be used to build the Keras Tuner model"
            },
            "hyperparameter_bank": {
                "type": "object",
                "description": "A map from profile name → hyperparameter profile.  Each profile has `tags`, `description`, `params`, `advice`, and `source`, containing a predefined profile for this subtype of model",
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
Old schema -> leads to needle in a haystack problem
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

"""
"***Generate and save a runnable Python code wrapped in <Code>…</Code> in a singe `notebook_code` json field:\\n"  
            "Your code must:\n\
                1. Seed Python, NumPy, scikit-learn, and TF/PyTorch for reproducibility.\n\
                2. Import on separate lines:\n\
                pandas\n\
                numpy\n\
                train_test_split\n\
                SimpleImputer\n\
                ColumnTransformer\n\
                StandardScaler\n\
                OneHotEncoder\n\
                Pipeline\n\
                tensorflow (or torch)\n\
                Sequential (from tensorflow.keras.models)\n\
                Model (from tensorflow.keras.models)\n\
                tf.keras.applications (e.g. ResNet50, VGG16)\n\
                EarlyStopping\n\
                ModelCheckpoint\n\
                json\n\
                time\n\
               3. Load and split data:\n\
                - Read `training_files`; infer/extract ID and `target_columns`; drop originals.\n\
                - If there is exactly one target column, use\n\
                `stratify=y_enc` in `train_test_split`.\n\
                - Otherwise (multi-column target), set `stratify=None` to avoid sklearn’s ValueError.\n\
                4. Encode target: single → LabelEncoder; multi → raw numpy array.\n\
                5. Build preprocessing pipeline:\n\
                - Numeric: median+indicator imputation → StandardScaler.\n\
                - Categorical: most_frequent imputation → OneHotEncoder(ignore unknown).\n\
                6. Define ANN with ≥2 hidden layers + Dropout/BatchNorm; compile with Adam, appropriate loss, and metrics.\n\
                7. Define model (choose one approach):\n\
                   - **Sequential**: stack layers via `Sequential([…])`.\n\
                   - **Functional API**: use `Model(inputs, outputs)`.\n\
                   - **Pretrained**: import from `tf.keras.applications` and fine-tune.\n\
                   - **Subclassing**: extend `tf.keras.Model` with `call()`.\n\
                   - **scikit-learn**: use classic estimators like `RandomForestClassifier`.\n\
                8. Compile with Adam, appropriate loss, and metrics.\n\
                9. Train with EarlyStopping(patience=5) and ModelCheckpoint(save_best_only), up to 100 epochs; record training time.\n\
                10. Log: to `results.json`\n\
                 - training loss\n\
                 - training accuracy\n\
                 - validation loss\n\
                 - validation accuracy\n\
                 - training duration \n\
                11. Predict & submit:\n\
                - **Load test.csv file; extract ID; preprocess; predict.\n\
                - Threshold or argmax as needed.\n\
                - Build DataFrame with ID first, targets next; save as `submission_result.csv` (no index).\n"
        
"""

""""***Generate and save a runnable Python code wrapped in <Code>…</Code> in a singe `notebook_code` json field:\\n"  
            "The generated code must implement:\n"
            "0. **Target encoding**: immediately after loading/concatenating train files, before split:\n"
            "   if len(target_columns) > 1:\n"
            "       # multilabel or multi-output regression\n"
            "       # keep shape (n_samples, n_targets)\n"
            "       if competition_problem_subtype in ['multilabel']:\n"
            "           y_enc = df[target_columns].astype(int).values\n"
            "       else:  # regression or quantile\n"
            "           y_enc = df[target_columns].astype(float).values\n"
            "   else:\n"
            "       # single-target classification or regression\n"
            "       col = target_columns[0]\n"
            "       if competition_problem_subtype in ['binary-classification','multiclass']:\n"
            "           from sklearn.preprocessing import LabelEncoder\n"
            "           le = LabelEncoder().fit(df[col].astype(str))\n"
            "           y_enc = le.transform(df[col].astype(str))\n"
            "           classes_ = le.classes_\n"
            "       else:\n"
            "           y_enc = df[col].astype(float).values\n"
            "   X = df.drop(columns=target_columns)  # drop targets before preprocessing\n"
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
            "3. **Data Loading & ID extraction**:\n"
            "- Read every file in `training_files` into `train_dfs`, and load `df_test` if a provided test file exists.\n"
            "- **Infer `id_col` & `target_columns`** from `submission_example`:\n"
            "```python\n"
            "header = submission_example.splitlines()[0]    # grab header row\n"
            "cols   = header.split(',')                    # split into column names\n"
            "id_col = cols[0]                              # first element is ID\n"
            "target_columns = cols[1:]                     # everything else are targets\n"
            "```\n"
            "- **Extract & drop ID**:\n"
            "```python\n"
            "# load and concatenate training files\n"
            "train_dfs = [pd.read_csv(f) for f in training_files]\n"
            "train_ids = pd.concat([df[id_col] for df in train_dfs], ignore_index=True)\n"
            "train_dfs = [df.drop(columns=[id_col]) for df in train_dfs]\n"
            "# handle provided test file if present\n"
            "if df_test is not None:\n"
            "    test_ids = df_test[id_col].copy()\n"
            "    df_test  = df_test.drop(columns=[id_col], errors='ignore')\n"
            "# now concatenate all train data into one df\n"
            "df = pd.concat(train_dfs, ignore_index=True)\n"
            "```\n"
            "4. **Feature Engineering**:\n"
            "- Automatically drop or transform obviously irrelevant columns, for example:\n"
            "```python\n"
            "# drop columns with all missing values\n"
            "df = df.dropna(axis=1, how='all')\n"
            "# drop categorical features with extremely high cardinality\n"
            "cat_cols = df.select_dtypes(include=['object']).columns\n"
            "high_card = [c for c in cat_cols if df[c].nunique() > 50]\n"
            "df = df.drop(columns=high_card)\n"
            "# (optional) apply simple feature transforms at your discretion\n"
            "```\n"
            "5. **Train/Validation Split**:\n"
            "- After target encoding, split into train/validation according to subtype:\n"
            "```python\n"
            "if competition_problem_subtype in ['time-series-forecasting','multivariate-time-series-forecasting']:\n"
            "    cutoff = int(len(X) * 0.8)\n"
            "    X_train, X_val = X[:cutoff], X[cutoff:]\n"
            "    y_train, y_val = y_enc[:cutoff], y_enc[cutoff:]\n"
            "else:\n"
            "    split_kwargs = dict(test_size=0.2, random_state=42)\n"
            "    if competition_problem_subtype in ['binary-classification','multiclass']:\n"
            "        split_kwargs['stratify'] = y_enc\n"
            "    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, **split_kwargs)\n"
            "```\n"
            "6. **Preprocessing Pipeline**:\n"
            "   - Auto-detect numeric vs. categorical via `df.select_dtypes`.\n"
            "   - Build a `ColumnTransformer` with median‐imputed & scaled numerics, and most‐frequent‐imputed & OHE categoricals (cap at 50 cats).\n"
            "   - Fit on train → transform train/val/test.\n"
            "7. **Fix numbering**: ensure your sections run 0→22 with no gaps or duplicates.\n"
            "8. **Model Definition**: two+ hidden layers, each with `BatchNormalization` & `Dropout`, final `units/activation` by subtype:\n"
            "   ```python\n"
            "   from tensorflow.keras.layers import Dense, BatchNormalization, Dropout\n"
            "   units = len(target_columns) if competition_problem_subtype in ['multilabel','multiclass'] else 1\n"
            "   if competition_problem_subtype == 'multiclass':\n"
            "       final_activation = 'softmax'\n"
            "   elif competition_problem_subtype in ['binary-classification','multilabel']:\n"
            "       final_activation = 'sigmoid'\n"
            "   else:\n"
            "       final_activation = 'linear'\n"
            "   # e.g. Sequential([... , Dense(units, activation=final_activation)])\n"
            "   ```\n"
            "9. **Compilation**: set `loss`/`metrics` by subtype:\n"
            "   ```python\n"
            "   if competition_problem_subtype == 'regression':\n"
            "       loss, metrics = 'mse', ['RootMeanSquaredError']\n"
            "   elif competition_problem_subtype == 'binary-classification':\n"
            "       loss, metrics = 'binary_crossentropy', ['accuracy']\n"
            "   elif competition_problem_subtype == 'multiclass':\n"
            "       loss, metrics = 'sparse_categorical_crossentropy', ['accuracy']\n"
            "   else:  # multilabel or multi-output regression\n"
            "       loss, metrics = 'binary_crossentropy' if competition_problem_subtype=='multilabel' else 'mse', ['accuracy'] if competition_problem_subtype=='multilabel' else ['RootMeanSquaredError']\n"
            "   model.compile(optimizer='adam', loss=loss, metrics=metrics)\n"
            "   ```\n"
            "10. **Callbacks & Training**: `EarlyStopping(monitor='val_loss', patience=5)` + `ModelCheckpoint(save_best_only=True)`, up to 100 epochs.\n"
            "11. **Evaluation & Logging**: load best weights, evaluate on train/val, save `training_accuracy`, `validation_accuracy`, losses & duration to `results.json`.\n"
            "12. **Prediction & Submission**: always one column per row:\n"
            "   ```python\n"
            "   preds = model.predict(X_test_proc)\n"
            "   if competition_problem_subtype == 'multiclass':\n"
            "       idx   = preds.argmax(axis=1)\n"
            "       final = le.inverse_transform(idx)\n"
            "   elif competition_problem_subtype == 'binary-classification':\n"
            "       final = (preds.flatten() > 0.5).astype(int)\n"
            "   else:\n"
            "       final = preds.flatten()\n"
            "   submission = pd.DataFrame({id_col: test_ids, target_columns[0]: final})\n"
            "   submission.to_csv('submission_result.csv', index=False)\n"
            "   ```\n"

            "13. Load the test data by reading the testing files into a DataFrame.\n"
            "14. Copy the ID column (`id_col`) from the test DataFrame and save it as `ids_test` (**never** drop the `id` column).\n"  
            "15. Apply the preprocessing pipeline (`preprocessor.transform`) to the feature-only DataFrame to produce `X_test_proc`.\n"  
            "16. Feed `X_test_proc` into your trained model to obtain raw predictions (`raw_preds`).\n"  
            "17. **Prediction & Submission**:\n"
            "   preds = model.predict(X_test_proc)\n"
            "   # single-target case\n"
            "   if len(target_columns) == 1:\n"
            "       if competition_problem_subtype == 'multiclass':\n"
            "           idx   = preds.argmax(axis=1)\n"
            "           final = le.inverse_transform(idx)\n"
            "       elif competition_problem_subtype == 'binary-classification':\n"
            "           final = (preds.flatten() > 0.5).astype(int)\n"
            "       else:  # regression\n"
            "           final = preds.flatten()\n"
            "       submission = pd.DataFrame({id_col: test_ids, target_columns[0]: final})\n"
            "   # multi-target case\n"
            "   else:\n"
            "       if competition_problem_subtype == 'multilabel':\n"
            "           final = (preds > 0.5).astype(int)\n"
            "       else:  # multi-output regression\n"
            "           final = preds\n"
            "       submission = pd.DataFrame(final, columns=target_columns)\n"
            "       submission[id_col] = test_ids\n"
            "       submission = submission[[id_col] + target_columns]\n"
            "   submission.to_csv('submission_result.csv', index=False)\n"

            "19. Reorder the submission DataFrame so that `ids_test` (under `id_col`) is the first column.\n"  
            "20. Write the submission DataFrame to `'submission_result.csv'` without row indices.\n"  
            "21. **Place this block at the end of your notebook**—after loading either the provided `'test.csv'` or your split DataFrame, predicting with `best_model`, and running these steps to generate and save the final CSV with the correct `id` and target column(s).**\n"  
            "22. **If multiple target columns exist, verify that each appears as its own column in `submission_result.csv`.**\n"  
"""


""""***Generate and save a runnable Python code wrapped in <Code>…</Code> in a singe `notebook_code` json field:\\n"  
            "The generated code must implement:\n"
            "0. **Target encoding**: after loading/concatenating train files & before any split:\n"
            "```python\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "col=target_columns[0]\n"
            "if competition_problem_subtype in ['binary-classification','multiclass']:\n"
            "    from sklearn.preprocessing import LabelEncoder"
            "    le = LabelEncoder().fit(df[col].astype(str))\n"
            "    y_enc = le.transform(df[col].astype(str)).astype(int)\n"
            "    classes_=le.classes_\n"
            "elif competition_problem_subtype=='multiclass-classification':\n"
            "    le=LabelEncoder().fit(df[col].astype(str))\n"
            "    y_enc=le.transform(df[col].astype(str))             # 0…K-1\n"
            "    classes_=le.classes_\n"
            "elif competition_problem_type=='regression' and len(target_columns)==1:\n"
            "    y_enc=df[col].astype(float).values                  # continuous\n"
            "else:\n"
            "    y_enc=df[target_columns].values                     # multi-output (multilabel/regression)\n"
            "X=df.drop(columns=target_columns, ignore_missing=True)\n"
            "```"
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
            "    1. Extract IDs and targets exactly by their raw CSV names, using a conditional pop/drop **(always include ignore_missing=True):\n"
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
            "12. **Prediction & Submission**:\n"
            "raw = model.predict(X_test_proc)\n"
            "if competition_problem_subtype == 'multiclass-classification':\n"
            "    preds = raw.argmax(axis=1)\n"
            "elif competition_problem_subtype in ['binary-classification','multilabel']:\n"
            "    preds = (raw > 0.5).astype(int)\n"
            "elif competition_problem_type == 'regression'\n"
            "    preds = raw.flatten()\n"
            "if len(target_columns) == 1:\n"
            "    submission = pd.DataFrame({id_col: test_ids, target_columns[0]: preds})\n"
            "else:\n"
            "    submission = pd.DataFrame(preds, columns=target_columns)\n"
            "    submission.insert(0, id_col, test_ids)\n"
            "df_out.insert(0, id_col, test_ids)\n"
            "df_out.to_csv('submission_result.csv', index=False)\n"
            "13. Load the test data by reading the testing files into a DataFrame.\n"
            "14. Copy the ID column (`id_col`) from the test DataFrame and save it as `ids_test` (**never** drop the `id` column).\n"  
            "15. Remove target columns but keep the ID column\n"
            "if using_provided_test_file then\n"
            "    # provided test.csv has no targets → safe no-op drop\n"
            "    features_df ← test_df.drop(columns=target_columns, ignore_missing=True)\n"
            "else\n"
            "    # split-from-single-file scenario → remove original target columns\n"
            "    features_df ← test_df.drop(columns=target_columns, ignore_missing=True)\n"
            "end if\n"
            "# ensure id_col remains in features_df at all times\n"
            "16. Apply the preprocessing pipeline (`preprocessor.transform`) to the feature-only DataFrame to produce `X_test_proc`.\n"  
            "17. Feed `X_test_proc` into your trained model to obtain raw predictions (`raw_preds`).\n"  
            "18. Build the submission DataFrame:\n"  
            "    – If there is **one** target column, threshold or map `raw_preds` directly into that column alongside `ids_test`.\n"  
            "    – If there are **multiple** target columns, convert `raw_preds` into a table with those column names, then append `ids_test` under `id_col`.\n"  
            "19. Reorder the submission DataFrame so that `ids_test` (under `id_col`) is the first column.\n"  
            "20. Write the submission DataFrame to `'submission_result.csv'` without row indices.\n"  
            "21. **Place this block at the end of your notebook**—after loading either the provided `'test.csv'` or your split DataFrame, predicting with `best_model`, and running these steps to generate and save the final CSV with the correct `id` and target column(s).**\n"  
            "22. **If multiple target columns exist, verify that each appears as its own column in `submission_result.csv`.**\n"  
"""