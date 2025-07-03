import numpy as np
import pandas as pd
import pickle
import faiss
from pathlib import Path


from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

from config import INDEX_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Build TF-IDF indices (notebook descriptions & competition descriptions)
# ─────────────────────────────────────────────────────────────────────────────

import pickle
from pathlib import Path

import faiss
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR

def build_index(
    df_structured: pd.DataFrame,
    model_name: str = "voidism/diffcse-roberta-base-sts"
):
    """
    1) Combine competition_problem_description + dataset_metadata per row.
    2) Encode texts via DiffCSE (SentenceTransformer).
    3) One-hot encode chosen categorical columns.
    4) L2-normalize and concatenate [text_embedding | one_hot_vector].
    5) Build and persist a FAISS inner-product index + all artifacts.
    """
    # ensure index directory exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1) Combine text fields ---
    df = df_structured.copy()
    for col in ["competition_problem_description", "dataset_metadata"]:
        df[col] = df.get(col, "").fillna("")

    df["combined_text"] = (
        df["competition_problem_description"].str.strip()
        + "  "
        + df["dataset_metadata"].str.strip()
    )

    # --- 2) Encode with SentenceTransformer (DiffCSE) ---
    s_model = SentenceTransformer(model_name)
    texts = df["combined_text"].tolist()
    text_embeddings = s_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # already normalized, but we’ll normalize again later
    )

    # --- 3) One-hot encode categorical columns ---
    cats = [
        "competition_problem_type",
        "competition_problem_subtype",
        "competition_dataset_type",
        "evaluation_metrics"
        # "framework",    # for debugging, not necessary
    ]
    # fill any missing
    for c in cats:
        df[c] = df.get(c, "Unknown").fillna("Unknown")

    ohe = OneHotEncoder(
        sparse_output=False,
        dtype=np.float32,
        handle_unknown="ignore"
    )
    cat_matrix = ohe.fit_transform(df[cats].values)

    # --- 4) Concatenate & normalize ---
    # ensure float32
    if text_embeddings.dtype != np.float32:
        text_embeddings = text_embeddings.astype(np.float32)

    combined_vectors = np.hstack([text_embeddings, cat_matrix])
    # L2-normalize for inner-product search
    faiss.normalize_L2(combined_vectors)

    # --- 5) Persist artifacts ---
    # 5.1) Model name
    (INDEX_DIR / "text_encoder_model_name.txt").write_text(model_name, encoding="utf-8")

    # 5.2) OneHotEncoder
    with open(INDEX_DIR / "onehot_encoder.pkl", "wb") as f:
        pickle.dump(ohe, f)

    # 5.3) Raw embeddings
    np.save(INDEX_DIR / "combined_embeddings.npy", combined_vectors)

    # 5.4) Build & save FAISS index
    dim = combined_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(combined_vectors)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.ip"))

    # 5.5) Row IDs
    row_ids = df["kernel_ref"].tolist()
    with open(INDEX_DIR / "row_ids.pkl", "wb") as f:
        pickle.dump(row_ids, f)

    print(f"[INFO] Saved model name → {INDEX_DIR/'text_encoder_model_name.txt'}")
    print(f"[INFO] Saved OHE → {INDEX_DIR/'onehot_encoder.pkl'}")
    print(f"[INFO] Saved embeddings → {INDEX_DIR/'combined_embeddings.npy'}")
    print(f"[INFO] Saved FAISS index → {INDEX_DIR/'faiss_index.ip'}")
    print(f"[INFO] Saved row IDs → {INDEX_DIR/'row_ids.pkl'}")
