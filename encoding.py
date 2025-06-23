import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

from config import INDEX_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Build TF-IDF indices (notebook descriptions & competition descriptions)
# ─────────────────────────────────────────────────────────────────────────────

def build_index(df_structured: pd.DataFrame,
                        model_name: str = "voidism/diffcse-roberta-base-sts"):
    """
    1) For each row, concatenate competition_problem_description,
    competition_dataset_description, and notebook_description into one string.
    2) Encode that combined text with a SentenceTransformer (DiffCSSE).
    3) One-hot encode the categorical columns:
    - competition_type
    - competition_problem_subtype
    - competition_dataset_type  (optional if always the same)
    - framework
    4) Concatenate the text embedding (e.g. 768-dim) with the one-hot vector.
    5) Save:
    - the SentenceTransformer model name (so we can reload it)
    - the OneHotEncoder object
    - the final N×D array of concatenated embeddings
    - the list of row IDs (e.g. kernel_ref) in the same order
    """

    INDEX_DIR.mkdir(exist_ok=True)

    # --- 1) Combine text fields per row ---
    # Assumes df_structured has these columns:
    #   competition_slug (or competition_id), competition_problem_description,
    #   competition_dataset_description, notebook_description,
    #   competition_type, competition_problem_subtype,
    #   competition_dataset_type, framework, kernel_ref (or notebook ID)

    # If some columns might be missing, verify or fillna("") before concatenating
    df = df_structured.copy()
    for col in ["competition_problem_description",
                "competition_dataset_description",
                #"notebook_description" - omit this for now, 
                # since a new given competition doesn't have a solution to compare to
                ]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    def _combine_text(row):
        parts = [
            row["competition_problem_description"].strip(),
            row["dataset_metadata"].strip(),
            #row["notebook_description"].strip(), also since we are using preprocessing NLP and code segments
        ]
        # join with a separator; extra spaces don't harm SimCSE
        return " ".join(p for p in parts if p)

    df["combined_text"] = df.apply(_combine_text, axis=1)

    # --- 2) Encode combined_text with SentenceTransformer ---
    s_model = SentenceTransformer(model_name)  # uses the saved model_name DiffCSE
    texts = df["combined_text"].tolist()
    text_embeddings = s_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # --- 3) One‐Hot Encode categorical columns ---
    # Choose the categorical fields we want to include:
    cats = ["competition_type",
            "competition_problem_subtype",
            "competition_dataset_type",
            #"framework" since we don't have a solution we can also omit this
            ]
    # Fill missing categories with a placeholder (e.g., "Unknown")
    for c in cats:
        if c not in df.columns:
            df[c] = "Unknown"
        else:
            df[c] = df[c].fillna("Unknown")

    # Fit a single OneHotEncoder on all four columns at once
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
    cat_matrix = ohe.fit_transform(df[cats].values)
    # cat_matrix.shape == (N, C_total), e.g., (10000, 10)

    # --- 4) Concatenate text embeddings + one-hot vectors ---
    # If text_embeddings is float64, cast to float32 to save space
    if text_embeddings.dtype != np.float32:
        text_embeddings = text_embeddings.astype(np.float32)

    # Concatenate along the last axis: [ text_embed | cat_onehot ]
    combined_vectors = np.hstack([text_embeddings, cat_matrix])
    # combined_vectors.shape == (N, H + C_total)

    # --- 5) Save everything needed for later queries ---
    # 5.1) Save the SentenceTransformer model name (not the full weights)
    with open(INDEX_DIR / "text_encoder_model_name.txt", "w") as f:
        f.write(model_name)

    # 5.2) Save the OneHotEncoder object
    with open(INDEX_DIR / "onehot_encoder.pkl", "wb") as f:
        pickle.dump(ohe, f)

    with open(INDEX_DIR / "onehot_categories.pkl", "wb") as f:
        pickle.dump(ohe.categories_, f)

    # 5.3) Save the combined_vectors matrix (N×D)
    # Depending on N, D, we might want to use np.save or pickle
    np.save(INDEX_DIR / "combined_embeddings.npy", combined_vectors)
    

    # 5.4) Save the list of row IDs (e.g. kernel_ref) in the same order
    row_ids = df["kernel_ref"].tolist()
    with open(INDEX_DIR / "row_ids.pkl", "wb") as f:
        pickle.dump(row_ids, f)

    print(f"[INFO] Saved text‐encoder model name into {INDEX_DIR/'text_encoder_model_name.txt'}")
    print(f"[INFO] Saved OneHotEncoder (categorical) to {INDEX_DIR/'onehot_encoder.pkl'}")
    print(f"[INFO] Saved combined {combined_vectors.shape} embeddings to {INDEX_DIR/'combined_embeddings.npy'}")
    print(f"[INFO] Saved {len(row_ids)} row IDs to {INDEX_DIR/'row_ids.pkl'}")
