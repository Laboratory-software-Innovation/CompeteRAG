# CompeteRAG

#### IMPORTANT: Please make sure to follow every point in the README.

Retrieval‑Augmented Generation pipeline that:

1. **Scrapes & structures** Kaggle competition pages, datasets and high‑quality TensorFlow / PyTorch notebooks.
2. **Embeds & indexes** metadata(DiffCSE sentence embeddings + weighted One Hot Encoding) in a weighted FAISS similarity index.
3. **Generates** reproducible Keras or Keras‑Tuner baseline notebooks for *new* competitions with the help of OpenAI GPT models.


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
		├──selenium_helper  # Default selenium tool (uses Chrome)
		├──selenium_firefox # Firefox version
		└──selenium_helper  # Edge version
```

---

## Quick start

### 1 · Clone & install

```bash
git clone https://github.com/IllyaGY/REU.git
cd REU
python -m venv venv && source venv/bin/activate   
venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Configure credentials

| What               | Where / how                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI key**     | `.env` → `OPENAI_API_KEY=sk‑…`                                                                                                                        |
| **Kaggle token**   | `~/.kaggle/kaggle.json` *or* set `KAGGLE_CONFIG_DIR`                                                                                                  |
| **Browser driver** | Pick one of `helpers/selenium_helper_firefox.py` or `…_edge.py`, rename to `helpers/selenium_helper.py`, ensure the matching driver binary is on PATH |

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

#### IMPORTANT: Before training, always download the files from Kaggle itself, do not rely on the files downloaded by the RAG for code generation, those may be unsupported due to how RAG extracts and decodes them. 

### 6 · Iterate with follow‑up prompts

If the first notebook fails, wrap the traceback inside `<Error> … </Error>` and the code inside `<Code> ... </Code>` run:

```bash
python rag.py followup 0 <slug>   # 0 = Keras, 1 = K‑T
```

### 7 · Iterate with the original prompt

Sometime, the followup may result in the same error over and over again, simply try running the original prompt. (Usually may be required when the dimensions are wrong)

```bash
python rag.py code 1|0 <slug>
```
---

## Execution flow (high‑level)

1. **rag.py (CLI)** — parses sub‑command
   - `cb` → `collection.collect_and_structured(max_notebooks = 5, start = None)` → `encoding.build_index()`
   - `b`  → `encoding.build_index()` (re‑index only)
   - `code` → `llm_coding.solve_competition_keras()` (uses `similarity.find_similar_ids()`) - Keras
   - `code` → `llm_coding.solve_competition_tuner()` - Keras Tuner
   - `followup` → `llm_coding.followup_prompt()`
1. **collection.py**
   - `collect_and_structured(max_notebooks = 5, start = None)` loops over slug lis
2. **encoding.py**
   - `build_index()` → DiffCSE embeds + OHE
     - `faiss.IndexFlatIP` saved to disk
4. **similarity.py**
   - `index.search(query_vec, k)` returns top‑k ids + scores
2. **llm\_coding.py**
   - `solve_competition_keras(slug, k, tuner_flag)`
     - calls `similarity.find_similar_ids()`
     - assembles prompt via `prompts.json_schema`
     - streams GPT and Deepseek-R1 function‑calls
   - `followup_prompt(slug, kt = 0|1)` same pipeline with error context


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

## Contributing & license



---



 