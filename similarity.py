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
) -> List[Tuple[str, float]]:
    """
    Given the path to a competition description JSON, return the top_k most-similar
    existing kernels (identified by their row_ids) along with similarity scores.

    Relies on the artifacts written by build_index():
      - INDEX_DIR/faiss_index.ip               — the FAISS Inner-Product index
      - INDEX_DIR/text_encoder_model_name.txt  — the DiffCSE model name
      - INDEX_DIR/onehot_encoder.pkl           — the fitted OneHotEncoder
      - INDEX_DIR/row_ids.pkl                  — list of kernel_refs in index order
    """

    # 1) load persisted artifacts
    index = faiss.read_index(str(INDEX_DIR / "faiss_index.ip"))
    row_ids = pickle.loads((INDEX_DIR / "row_ids.pkl").read_bytes())
    ohe     = pickle.loads((INDEX_DIR / "onehot_encoder.pkl").read_bytes())
    model_name = (INDEX_DIR / "text_encoder_model_name.txt").read_text().strip()
    s_model = SentenceTransformer(model_name)

    # 2) load this competition’s metadata
    meta = json.loads(Path(desc_json).read_text(encoding="utf-8"))
    prob_desc = meta.get("competition_problem_description", "")
    data_desc = meta.get("dataset_metadata", "")
    # ensure strings
    if not isinstance(prob_desc, str):
        prob_desc = json.dumps(prob_desc, ensure_ascii=False)
    if not isinstance(data_desc, str):
        data_desc = json.dumps(data_desc, ensure_ascii=False)

    # 3) encode text
    combined_text = prob_desc.strip() + "  " + data_desc.strip()
    text_vec = s_model.encode(
        [combined_text],
        normalize_embeddings=True
    )[0].astype(np.float32)

    # 4) one-hot encode categories
    cats = [
        meta.get("competition_problem_type", "Unknown"),
        meta.get("competition_problem_subtype", "Unknown"),
        meta.get("competition_dataset_type", "Unknown"),
    ]
    cat_vec = ohe.transform([cats])[0].astype(np.float32)

    # 5) build final query vector and L2-normalize
    qv = np.concatenate([text_vec, cat_vec], axis=0)
    faiss.normalize_L2(qv.reshape(1, -1))

    # 6) search
    D, I = index.search(qv.reshape(1, -1), top_k + (1 if exclude_competition else 0))

    # 7) collect results (skipping any excluded slug)
    results: List[Tuple[str, float]] = []
    for idx, score in zip(I[0], D[0]):
        rid = row_ids[idx]
        if exclude_competition and rid.startswith(exclude_competition):
            continue
        results.append((rid, float(score)))
        if len(results) >= top_k:
            break

    return results


