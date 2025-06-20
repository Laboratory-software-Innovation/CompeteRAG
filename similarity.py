import json
import pickle
import numpy as np
import faiss
from pathlib import Path

from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from config import INDEX_DIR


def find_similar_ids(
    desc_json: str,
    top_k: int = 5,
    exclude_competition: str = None
) -> List[Tuple[str,float]]:
    """
    Same as find_similar_from_description but *returns* a list of
    (kernel_ref, score) instead of printing.
    """
    # ── load index artifacts ────────────────────────────
    combined = np.load(INDEX_DIR / "combined_embeddings.npy")
    with open(INDEX_DIR / "row_ids.pkl","rb") as f: row_ids = pickle.load(f)
    with open(INDEX_DIR / "onehot_encoder.pkl","rb") as f: ohe = pickle.load(f)

    # ── reload text‐encoder ──────────────────────────────
    model_name = (INDEX_DIR/"text_encoder_model_name.txt").read_text().strip()
    s_model = SentenceTransformer(model_name)

    # ── build query_vec just like before ────────────────
    meta = json.loads(Path(desc_json).read_text())

    # Safely extract descriptions, stringify non-strings
    prob_desc = meta.get("competition_problem_description", "")
    data_desc = meta.get("competition_dataset_description", "")
    if not isinstance(prob_desc, str):
        prob_desc = json.dumps(prob_desc, ensure_ascii=False)
    if not isinstance(data_desc, str):
        data_desc = json.dumps(data_desc, ensure_ascii=False)

    combined_text = prob_desc.strip() + "  " + data_desc.strip()
    text_vec = s_model.encode([combined_text], normalize_embeddings=True)[0]

    # Not used for now
    # combined_text = meta["competition_problem_description"].strip() + "  " + meta["competition_dataset_description"].strip()
    # text_vec = s_model.encode([combined_text], normalize_embeddings=True)[0]

    cat_vals = [meta[c] for c in ["competition_type","competition_problem_subtype","competition_dataset_type"]]
    cat_vec  = ohe.transform([cat_vals])[0]
    qv = np.concatenate([text_vec, cat_vec], axis=0).astype("float32")

    # ── normalize & search ──────────────────────────────
    faiss.normalize_L2(combined)
    faiss.normalize_L2(qv.reshape(1,-1))
    index = faiss.IndexHNSWFlat(combined.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(combined)

    # top-k
    D, I = index.search(qv.reshape(1,-1), top_k + (1 if exclude_competition else 0))
    results = []
    for idx, score in zip(I[0], D[0]):
        rid = row_ids[idx]
        if exclude_competition and rid.startswith(exclude_competition):
            continue
        results.append((rid, float(score)))
        if len(results) >= top_k:
            break

        
    return results



