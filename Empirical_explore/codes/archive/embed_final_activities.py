#!/usr/bin/env python3
"""Embed final valid normalized activity phrases with a reproducible row map."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "phrases/clean_normalize/activities_medium_final_complete.csv"
OUTPUT_DIR = PROJECT / "phrases/clean_normalize"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def main() -> None:
    frame = pd.read_csv(INPUT)
    valid = frame.loc[frame["final_validation"].eq("valid")].copy()
    valid = valid.reset_index(drop=True)
    if valid["activity_phrase_clean"].isna().any():
        raise ValueError("Valid rows contain missing activity_phrase_clean values")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        valid["activity_phrase_clean"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if embeddings.shape[0] != len(valid):
        raise RuntimeError("Embedding count does not match valid phrase count")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Embeddings are not unit normalized")

    np.save(OUTPUT_DIR / "activity_embeddings_medium_final.npy", embeddings)
    mapping = valid[[
        "row_id", "firm_id", "ticker", "year", "activity_phrase_raw",
        "activity_phrase_clean", "source_sentence", "source_method",
    ]].copy()
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=int))
    mapping["embedding_model"] = MODEL_NAME
    mapping.to_csv(
        OUTPUT_DIR / "activity_embeddings_medium_final_index.csv", index=False
    )
    print(f"Embedded rows: {len(valid)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
