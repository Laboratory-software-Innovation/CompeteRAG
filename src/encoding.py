import numpy as np
import pandas as pd
import pickle
import faiss
from pathlib import Path
from collections import OrderedDict


from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

from config import INDEX_DIR

import pickle
from pathlib import Path

import faiss
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR

# Build TF-IDF indices (notebook descriptions & competition descriptions)
def build_index(
    df_structured: pd.DataFrame,
    model_name: str = "voidism/diffcse-roberta-base-sts",
    cat_weights: dict = {
        "competition_problem_subtype": 3.0,
        "competition_problem_type":      2.5,
        "evaluation_metrics":           2.0,
        "competition_dataset_type":     1.0
    }
):
    """
    1) Combine competition_problem_description + dataset_metadata.
    2) Encode text via SentenceTransformer.
    3) One-hot encode each categorical column separately, scaling by its weight.
    4) Concatenate [text_emb | weighted_ohe] and L2-normalize.
    5) Persist all artifacts and build a FAISS IndexFlatIP.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = df_structured.copy()
    for col in ["competition_problem_description", "dataset_metadata"]:
        df[col] = df.get(col, "").fillna("")
    df["combined_text"] = df["competition_problem_description"].str.strip() + "  " + df["dataset_metadata"].str.strip()

    # 2) Text embeddings
    s_model = SentenceTransformer(model_name)
    text_embs = s_model.encode(
        df["combined_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # 3) Per-column OHE with weights
    cat_weights = OrderedDict(cat_weights)
    ohe_dict = OrderedDict()
    cat_parts = []
    for col, weight in cat_weights.items():
        df[col] = df.get(col, "Unknown").fillna("Unknown")
        ohe = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
        enc = ohe.fit_transform(df[[col]]) * weight
        ohe_dict[col] = ohe
        cat_parts.append(enc)

    cat_mat = np.hstack(cat_parts)

    # 4) Combine & normalize
    vectors = np.hstack([text_embs, cat_mat])
    faiss.normalize_L2(vectors)

    # 5) Persist artifacts
    (INDEX_DIR / "text_encoder_model_name.txt").write_text(model_name, encoding="utf-8")
    np.save(INDEX_DIR / "combined_embeddings.npy", vectors)
    pickle.dump(df["kernel_ref"].tolist(),    (INDEX_DIR / "row_ids.pkl").open("wb"))
    pickle.dump(ohe_dict,                     (INDEX_DIR / "onehot_encoder.pkl").open("wb"))
    pickle.dump(cat_weights,                  (INDEX_DIR / "cat_weights.pkl").open("wb"))

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.ip"))

    print(f"[INFO] Saved model name → {INDEX_DIR/'text_encoder_model_name.txt'}")
    print(f"[INFO] Saved OHE → {INDEX_DIR/'onehot_encoder.pkl'}")
    print(f"[INFO] Saved embeddings → {INDEX_DIR/'combined_embeddings.npy'}")
    print(f"[INFO] Saved FAISS index → {INDEX_DIR/'faiss_index.ip'}")
    print(f"[INFO] Saved row IDs → {INDEX_DIR/'row_ids.pkl'}")
