# CompeteRAG

#### IMPORTANT: Please make sure to follow every point in the README.

Retrieval‑Augmented Generation pipeline that:

1. **Scrapes & structures** Kaggle competition pages, datasets and high‑quality TensorFlow / PyTorch notebooks.
2. **Embeds & indexes** metadata(RoBERTa + DiffCSE sentence embeddings + weighted One Hot Encoding) in a weighted FAISS similarity index.
3. **Generates** reproducible Keras or Keras‑Tuner baseline notebooks for *new* competitions with the help of OpenAI GPT models.


## Configuration used

Python 3.12.3

Ubuntu 22.04

Chrome

GPT o4-mini and Deepseek-R1

---

## Repository layout

```
├──rag.py
└──src/
	├── collection.py      # Scrape competitions + notebooks → structured CSV
	├── encoding.py        # Build & save FAISS index of competitions/notebooks
	├── llm_coding.py      # Generate baseline solutions (Keras / Keras‑Tuner)
	├── prompts.py         # JSON schemas for GPT function calls
	├── similarity.py      # k‑NN search over FAISS index
	├── tuner_bank.py      # Pre‑built hyper‑parameter search spaces
	├── utils.py           # HTML scraping, file extraction, schema summarisation
	├── config.py          # Paths, constants, Kaggle & OpenAI setup
	├── rag.py             # **CLI entry‑point** (collect ▶ build ▶ code ▶ follow‑up)
	├── requirements.txt   # All Python dependencies
	└── helper/
		├──selenium_helper  # Selenium tool (uses Chrome)
```

---

## Quick start

### 1 · Clone & install

```bash
git clone https://github.com/Laboratory-software-Innovation/CompeteRAG
cd CompeteRAG
pip install -r requirements.txt
sudo pip install kaggle
chmod 600 /home/<user>/.config/kaggle/kaggle.json
```
Ensure Kaggle API is installed with sudo and the kaggle.json file has the permission set to 600


### 2 · Configure credentials

| What               | Where / how                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI key**     | `.env` → `OPENAI_API_KEY=sk‑…`                                                                                                                        |
| **Kaggle token**   | `~/.config/kaggle/kaggle.json` *or* set `KAGGLE_CONFIG_DIR`                                                                                                  |

### 3 · Collect notebooks & build index

```bash
python rag.py cb                 # all competitions in comps.py
python rag.py cb <slug>          # start from a specific competition
```

Outputs:

- `notebooks_structured.csv`
- `index_data/faiss_index.ip`

### 4 · Generate code for a new competition

#### IMPORTANT: Before generating the code, ensure you have joined the competition you would like to generate code for.

```bash
# Outline
python3 rag.py code <keras-tuner 0|1> <slug> <top-k: 1-9> 

# Standard Keras 

#(top-1 similar notebook) starting from the very beginning
python3 rag.py code 0 

#(top-5 similar notebook) starting from the very beginning
python3 rag.py code 0 5

#(top‑k similar notebooks = 3) starting(and including) from a certain competition
python rag.py code 0 <slug> 3

#Similar applies to Keras Tuner. However the top-k number doesn't apply in this case
python rag.py code 1 

# Keras‑Tuner version built from the generated Keras notebook
python rag.py code 1 <slug>
```

Creates under `test/<slug>/`:

- `<slug>_desc.json` – compact competition description
- `<slug>_solution.py`  or  `<slug>_kt_solution.py`

### 5 Running the model

The generated python file will be placed in test/`<slug>`, the code itself will appear in `<Code>...</Code>` make sure to remove them before running it. Also the make sure you specify the file paths manually since the model simply input the file names and due to how RAG loads data files, it might have different extension for those files.

#### IMPORTANT: 

##### 1) Before training, always download the files from Kaggle itself, do not rely on the files downloaded by the RAG for code generation, those may be unsupported due to how RAG extracts and decodes them. 
##### 2) Verify that the metrics used are correct (Accuracy - Classification, RMSE - Regression). Sometimes, the metrics used by the LLM are incorrect.
###### For example, due to how the prompt is structured, the model may generate a log1p RMSE and MAE scores. We used these metrics instead:

```python
def mse_real(y_true_log, y_pred_log):
	y_true = tf.math.expm1(y_true_log)
	y_pred = tf.math.expm1(y_pred_log)
	return tf.reduce_mean(tf.square(y_true - y_pred))
mse_real.__name__ = 'mse_real'

  
def rmse_real(y_true_log, y_pred_log):
	y_true = tf.math.expm1(y_true_log)
	y_pred = tf.math.expm1(y_pred_log)
	return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
rmse_real.__name__ = 'rmse_real'
```




### 6 · Iterate with follow‑up prompts

If the first notebook fails, wrap the traceback inside `<Error> … </Error>` and the code inside `<Code> ... </Code>` run:

```bash
python rag.py followup 0 <slug>   # 0 = Keras, 1 = K‑T
```

### 7 · Re-generate with the original prompt

Sometime, the followup may result in the same error over and over again, simply try running the original prompt. (Usually may be required when the dimensions are wrong)

```bash
python rag.py code 1|0 <slug>
```
---

## Execution flow (high‑level)

1. **rag.py (CLI)** — parses sub‑command\
   • `cb`   → `collection.collect_and_structured()` → `encoding.build_index()`\
   • `b`    → `encoding.build_index()` (re‑index only)\
   • `code` → `llm_coding.solve_competition_keras()` (*Keras*) or `llm_coding.solve_competition_tuner()` (*K‑T*)\
   • `followup` → `llm_coding.followup_prompt()`

2. **collection.py** — `collect_and_structured()` loops over `train` list and downloads notebooks

3. **encoding.py** — `build_index()` → DiffCSE embeddings + weighted OHE → saves `faiss.IndexFlatIP`

4. **similarity.py** — `find_similar_ids()` queries FAISS index

5. **llm\_coding.py** — orchestrates prompts, builds code via GPT, handles follow‑ups
	- `solve_competition_keras()`
     - calls `similarity.find_similar_ids()` to find top-k similar examples
     - assembles prompt via `prompts.py` json schema
     - streams GPT-o4-mini and Deepseek-R1 function‑calls


## Module overview

| File            | Purpose                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| `collection.py` | HTML + Kaggle API scraping, notebook filtering, LLM‑based structuring                                   |
| `encoding.py`   | Sentence‑Transformer + weighted OHE → FAISS index build & persist                                       |
| `similarity.py` | Query helper that returns top‑k similar kernel refs for a given competition JSON                        |
| `llm_coding.py` | High‑level orchestration → builds data profiles, selects examples, calls GPT tools, post‑processes code |
| `prompts.py`    | All JSON schemas fed to GPT (labeling, structuring, code generation)                                    |
| `tuner_bank.py` | Library of hyper‑parameter spaces for Keras‑Tuner                                                       |
| `utils.py`      | Selenium helpers, HTML parsing, dataset archive extraction, schema compaction                           |
| `config.py`     | Constants (paths, weights, max features/classes), env loading, authenticated KaggleApi instance         |
| `rag.py`        | Command‑line interface and glue code                                                                    |

---

## Troubleshooting

- **HTTP 403 from Kaggle** → ensure you have *joined* the competition and your token is valid.
- **Selenium **`` → browser and driver versions mismatch.
- ``** fails** → run `python rag.py cb` or `rag.py b` to (re)build the index.
- **GPT/tool call timeouts** → reduce `top_k`, or retry.

---

## Acknowledgements

This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.

---



 
