# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute Similarities (each competition vs. all notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_from_description(
    desc_json: str,
    top_k: int = 5,
    exclude_competition: str = None
) -> None:
    """
    desc_json: path to a JSON file with exactly these keys:
      {
        "competition_problem_description": str,
        "competition_dataset_description": str,
        "competition_type": str,
        "competition_problem_subtype": str,
        "competition_dataset_type": str,
        "framework": str
      }
    """

    # ── 1) Load index artifacts ──────────────────────────────────────────
    model_name = (INDEX_DIR / "text_encoder_model_name.txt").read_text().strip()
    combined = np.load(INDEX_DIR / "combined_embeddings.npy")  # shape (N, D_total)
    with open(INDEX_DIR / "row_ids.pkl", "rb") as f:
        row_ids = pickle.load(f)                                # list of length N
    with open(INDEX_DIR / "onehot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)                                    # OneHotEncoder

    # compute dims
    C_total = sum(len(cats) for cats in ohe.categories_)
    text_dim = combined.shape[1] - C_total
    D_total = combined.shape[1]

    # ── 2) Reload the exact same SentenceTransformer ──────────────────
    s_model = SentenceTransformer(model_name)

    # ── 3) Read & combine new competition fields ────────────────────
    meta = json.loads(Path(desc_json).read_text(encoding="utf-8"))
    # 3a) text
    combined_text = " ".join([
        meta["competition_problem_description"].strip(),
        meta["competition_dataset_description"].strip()
    ])
    # 3b) cats
    cats_order = [
        "competition_type",
        "competition_problem_subtype",
        "competition_dataset_type",
        #"framework"
    ]
    cat_vals = [ meta[c] for c in cats_order ]

    # ── 4) Encode new query vector ──────────────────────────────────
    text_vec = s_model.encode(
        [combined_text],
        normalize_embeddings=True
    )[0]  # shape (text_dim,)
    cat_vec = ohe.transform([cat_vals])[0]  # shape (C_total,)
    query_vec = np.concatenate([text_vec, cat_vec], axis=0).astype("float32")  # (D_total,)

    # ── 5) Build & query FAISS HNSW index with cosine (inner‐product) ───
    # Make sure the vectors are unit‐length so inner‐product = cosine
    faiss.normalize_L2(combined)                              # (N, D_total)
    faiss.normalize_L2(query_vec.reshape(1, -1))              # (1, D_total)

    # Use METRIC_INNER_PRODUCT instead of the default METRIC_L2
    index = faiss.IndexHNSWFlat(D_total, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch       = 64

    index.add(combined)                                        # add all N normalized vectors
    buffer = 10 
    # Now D will be the cosine‐similarity in [-1,1], not an L2 distance
    D, I = index.search(query_vec.reshape(1, -1), top_k + buffer + (1 if exclude_competition else 0))
    indices = I[0]
    scores  = D[0]


    # ── 6) Optionally filter out same competition ───────────────────
    results = []
    if exclude_competition:
        # load structured CSV to map row_id→competition_slug
        df_nb = pd.read_csv("notebooks_structured.csv", usecols=["kernel_ref","competition_slug"])
        slug_map = dict(zip(df_nb["kernel_ref"], df_nb["competition_slug"]))
        for idx, score in zip(indices, scores):
            rid = row_ids[idx]
            if len(results) >= top_k:
                break
            if slug_map.get(rid) == exclude_competition:
                continue
            results.append((rid, float(score)))
            
    else:
        results = [(row_ids[idx], float(score)) for idx, score in zip(indices, scores)]

    # ── 7) Print top_k ───────────────────────────────────────────────
    print(f"\nTop {top_k} matches (excluding='{exclude_competition}'):\n")
    for rank, (rid, sc) in enumerate(results[:top_k], start=1):
        print(f"{rank:2d}. {rid}   (score={sc:.4f})")
    print()











def solve_competition_with_examples(
    desc_json: str,
    train_csv_url: str,
    class_col: str,
    structured_csv: str = "notebooks_structured.csv",
    top_k: int = 5
) -> Dict[str,Any]:
    """
    1. Find the top_k similar notebooks.
    2. Download & profile train.csv → dataset_schema.
    3. Build & send a final LLM prompt with:
       - New comp metadata + dataset_schema
       - Summaries of those K notebooks
       - Instruction to propose a JSON solution outline
    """
    # 1) get top-k notebook refs + scores
    topk = find_similar_ids(desc_json, top_k=top_k, exclude_competition=None)

    # 2) load your structured outputs to fetch summaries
    df = pd.read_csv(structured_csv)
    example_texts = []
    for rank, (rid, score) in enumerate(topk, start=1):
        row = df[df["kernel_ref"] == rid].iloc[0]
        example_texts.append(
            f"{rank}. `{rid}` (score={score:.3f}): preprocessing={row['preprocessing_steps']}; "
            f"layers:\n{row['notebook_model_layers_code']}"
        )

    # 3) profile the new dataset
    schema_profile = describe_schema(train_csv_url, class_col)

    # 4) load new competition metadata
    new_meta = json.loads(Path(desc_json).read_text())

    # 5) build LLM messages
    system = {
        "role": "system",
        "content": (
            "You are an expert data scientist. "
            "Below is a new Kaggle competition and several example solutions. "
            "Using only these examples and the dataset schema, propose a complete modeling plan."
        )
    }
    user_new = {
        "role": "user",
        "content": (
            "New competition metadata:\n" +
            json.dumps(new_meta, ensure_ascii=False, indent=2) +
            "\n\nDataset schema:\n" +
            json.dumps(schema_profile, ensure_ascii=False, indent=2)
        )
    }
    user_examples = {
        "role": "user",
        "content": "Here are the top-K example notebooks:\n" + "\n".join(example_texts)
    }
    user_task = {
        "role": "user",
        "content": (
            "Based on the above examples and schema, outline in JSON:\n"
            "{\n"
            '  "architecture": "<model & rationale>",\n'
            '  "preprocessing": "<list of steps>",\n'
            '  "hyperparameters": "<learning rate, batch size, epochs, etc.>",\n'
            '  "extras": "<data aug, loss, training tricks>"\n'
            "}\n"
            "**Return only valid JSON.**"
        )
    }

    messages = [system, user_new, user_examples, user_task]

    # 6) call the LLM
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=messages
    )
    content = resp.choices[0].message.content.strip()

    # strip fences if any
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1]).strip()

    return json.loads(content)
