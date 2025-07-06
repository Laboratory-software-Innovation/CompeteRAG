import json
import pickle
import numpy as np
import faiss
from pathlib import Path
import pandas as pd 

from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from config import INDEX_DIR


def find_similar_ids(
    desc_json: str,
    top_k: int = 5,
    exclude_competition: str = None,
    text_weight: float = 1.0
) -> list[tuple[str,float]]:
    """
    Load the exact encoders & weights from build_index, encode a new
    competition’s text + cats, and return top_k most-similar row_ids.
    """
    # load index + row_ids
    index     = faiss.read_index(str(INDEX_DIR / "faiss_index.ip"))
    row_ids   = pickle.load((INDEX_DIR / "row_ids.pkl").open("rb"))
    ohe_dict  = pickle.load((INDEX_DIR / "onehot_encoder.pkl").open("rb"))
    cat_weights = pickle.load((INDEX_DIR / "cat_weights.pkl").open("rb"))
    model_name = (INDEX_DIR / "text_encoder_model_name.txt").read_text().strip()
    s_model    = SentenceTransformer(model_name)

    # load new metadata
    meta      = json.loads(Path(desc_json).read_text(encoding="utf-8"))
    prob_desc = meta.get("competition_problem_description","")
    data_desc = meta.get("dataset_metadata","")
    for txt in (prob_desc, data_desc):
        if not isinstance(txt, str):
            txt = json.dumps(txt, ensure_ascii=False)
    combined_text = prob_desc.strip() + "  " + data_desc.strip()

    # encode text
    text_vec = s_model.encode([combined_text], normalize_embeddings=True)[0].astype(np.float32)
    text_vec *= text_weight

    # encode cats
    cat_fields = list(cat_weights.keys())
    parts = []
    for col in cat_fields:
        val = meta.get(col, "Unknown")
        df_val = pd.DataFrame([{col: val}])
        enc = ohe_dict[col].transform(df_val).astype(np.float32)
        parts.append(enc * cat_weights[col])

    cat_vec = np.hstack(parts)[0]  # flatten to 1d

    # final query
    qv = np.concatenate([text_vec, cat_vec], axis=0).reshape(1,-1)
    faiss.normalize_L2(qv)

    # search
    D,I = index.search(qv, top_k + (1 if exclude_competition else 0))

    # collect
    results = []
    for idx,score in zip(I[0], D[0]):
        rid = row_ids[idx]
        if exclude_competition and rid.startswith(exclude_competition):
            continue
        results.append((rid, float(score)))
        if len(results) >= top_k:
            break
    return results