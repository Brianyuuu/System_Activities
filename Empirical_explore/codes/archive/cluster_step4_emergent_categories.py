#!/usr/bin/env python3
"""Step 4 only: cluster final firm-only activity phrases into emergent categories.

This script intentionally stops at clustering. It does not run Step 5 taxonomy,
theory mapping, or sanity-check work.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
INPUT = (
    PROJECT_DIR
    / "phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv"
)
EMBEDDING_DIR = PROJECT_DIR / "phrases/clean_normalize/embedding"
EXISTING_EMBEDDINGS = EMBEDDING_DIR / "firm_only_deduplicated_mpnet_embeddings.npy"
OUTPUT_DIR = PROJECT_DIR / "phrases/clean_normalize/clustering"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# The candidate range is deliberately broad enough to reveal coarse vs granular
# activity categories, while avoiding many tiny clusters from a 522-row corpus.
CANDIDATE_K = list(range(8, 33, 2))
PRIMARY_K = 24
PRIMARY_METHOD = "kmeans_embedding"
STABILITY_SEEDS = [11, 23, 37, 49, 61]


LABEL_STOPWORDS = {
    "activity",
    "activities",
    "company",
    "firm",
    "business",
    "service",
    "provide",
    "include",
    "including",
    "continue",
    "use",
    "make",
    "customer",
    "customers",
}


def load_or_create_embeddings(frame: pd.DataFrame) -> np.ndarray:
    if EXISTING_EMBEDDINGS.exists():
        embeddings = np.load(EXISTING_EMBEDDINGS)
        if embeddings.shape[0] == len(frame):
            return embeddings.astype(np.float32)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        frame["activity_phrase_clean"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embeddings


def threshold_name(k: int) -> str:
    return f"k{k:02d}"


def top_terms(phrases: list[str], n: int = 8) -> str:
    if not phrases:
        return ""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
    )
    matrix = vectorizer.fit_transform(phrases)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    names = np.array(vectorizer.get_feature_names_out())
    keep = [
        i
        for i, term in enumerate(names)
        if not all(piece in LABEL_STOPWORDS for piece in re.split(r"\s+", term))
    ]
    if keep:
        scores = scores[keep]
        names = names[keep]
    top = names[np.argsort(scores)[::-1][:n]]
    return ", ".join(top.tolist())


def suggested_label(phrases: list[str]) -> str:
    terms = top_terms(phrases, 4)
    if not terms:
        return "miscellaneous activity cluster"
    return terms


def agglomerative_labels(distance: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:  # Older sklearn fallback.
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance)


def kmeans_labels(embeddings: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(embeddings)


def ordered_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber arbitrary cluster labels into stable descending-size order."""

    sizes = pd.Series(labels).value_counts()
    first_row = {label: int(np.where(labels == label)[0][0]) for label in np.unique(labels)}
    ordered = sorted(np.unique(labels), key=lambda label: (-int(sizes[label]), first_row[label]))
    mapping = {old: new for new, old in enumerate(ordered)}
    return np.array([mapping[label] for label in labels], dtype=int)


def evaluate_k(frame: pd.DataFrame, embeddings: np.ndarray, distance: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in CANDIDATE_K:
        labels = agglomerative_labels(distance, k)
        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "method": "agglomerative_average_cosine",
                "k": k,
                "silhouette_cosine": float(silhouette_score(distance, labels, metric="precomputed")),
                "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels)),
                "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
                "min_cluster_size": int(sizes.min()),
                "median_cluster_size": float(sizes.median()),
                "max_cluster_size": int(sizes.max()),
                "singleton_clusters": int((sizes == 1).sum()),
                "clusters_size_lt_5": int((sizes < 5).sum()),
            }
        )

    for k in CANDIDATE_K:
        labels_by_seed = [
            KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(embeddings)
            for seed in STABILITY_SEEDS
        ]
        silhouettes = [
            silhouette_score(distance, labels, metric="precomputed") for labels in labels_by_seed
        ]
        sizes = pd.Series(labels_by_seed[0]).value_counts()
        rows.append(
            {
                "method": "kmeans_embedding",
                "k": k,
                "silhouette_cosine": float(np.mean(silhouettes)),
                "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels_by_seed[0])),
                "davies_bouldin": float(davies_bouldin_score(embeddings, labels_by_seed[0])),
                "min_cluster_size": int(sizes.min()),
                "median_cluster_size": float(sizes.median()),
                "max_cluster_size": int(sizes.max()),
                "singleton_clusters": int((sizes == 1).sum()),
                "clusters_size_lt_5": int((sizes < 5).sum()),
            }
        )
    return pd.DataFrame(rows)


def cluster_summary(
    frame: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray, method: str, k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = frame.copy()
    assignments.insert(0, "activity_cluster_id", [f"C{label + 1:02d}" for label in labels])
    assignments.insert(1, "activity_cluster_label_numeric", labels + 1)
    assignments.insert(2, "clustering_method", method)
    assignments.insert(3, "clustering_k", k)
    assignments["embedding_model"] = MODEL_NAME

    summary_rows = []
    rep_rows = []
    for label in sorted(np.unique(labels)):
        idx = np.where(labels == label)[0]
        sub = frame.iloc[idx]
        sub_embeddings = embeddings[idx]
        centroid = sub_embeddings.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = sub_embeddings @ centroid
        order = np.argsort(sims)[::-1]
        phrases = sub["activity_phrase_clean"].astype(str).tolist()
        cluster_id = f"C{label + 1:02d}"
        firms = sorted(sub["ticker"].astype(str).unique())
        source_methods = sorted(sub["source_method"].astype(str).unique())
        pairwise = cosine_similarity(sub_embeddings) if len(sub_embeddings) > 1 else np.array([[1.0]])
        within = pairwise[np.triu_indices_from(pairwise, k=1)] if len(sub_embeddings) > 1 else np.array([])
        summary_rows.append(
            {
                "activity_cluster_id": cluster_id,
                "activity_cluster_label_numeric": label + 1,
                "suggested_emergent_category_label": suggested_label(phrases),
                "top_terms": top_terms(phrases, 10),
                "cluster_size": len(sub),
                "n_firms": len(firms),
                "firms": "|".join(firms),
                "source_methods": "|".join(source_methods),
                "mean_similarity_to_centroid": float(np.mean(sims)),
                "min_similarity_to_centroid": float(np.min(sims)),
                "mean_within_cluster_pairwise_cosine": float(np.mean(within)) if len(within) else np.nan,
                "min_within_cluster_pairwise_cosine": float(np.min(within)) if len(within) else np.nan,
                "representative_phrase_1": sub.iloc[int(order[0])]["activity_phrase_clean"],
                "representative_phrase_2": sub.iloc[int(order[1])]["activity_phrase_clean"]
                if len(order) > 1
                else "",
                "representative_phrase_3": sub.iloc[int(order[2])]["activity_phrase_clean"]
                if len(order) > 2
                else "",
            }
        )
        for rank, local_pos in enumerate(order[:10], start=1):
            row = sub.iloc[int(local_pos)]
            rep_rows.append(
                {
                    "activity_cluster_id": cluster_id,
                    "rank": rank,
                    "similarity_to_centroid": float(sims[int(local_pos)]),
                    "row_id": row.get("row_id", ""),
                    "ticker": row.get("ticker", ""),
                    "company": row.get("company", ""),
                    "activity_phrase_clean": row.get("activity_phrase_clean", ""),
                    "activity_phrase": row.get("activity_phrase", ""),
                    "source_sentence": row.get("source_sentence", ""),
                    "source_method": row.get("source_method", ""),
                }
            )
    return assignments, pd.DataFrame(summary_rows), pd.DataFrame(rep_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(INPUT)
    if len(frame) != 522:
        raise ValueError(f"Expected 522 final phrases, found {len(frame)}")
    if not frame["actor_class"].eq("firm").all():
        raise ValueError("Input contains non-firm actor rows")

    embeddings = load_or_create_embeddings(frame)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        embeddings = embeddings / np.maximum(norms[:, None], 1e-12)

    distance = pairwise_distances(embeddings, metric="cosine")
    model_selection = evaluate_k(frame, embeddings, distance)
    model_selection.to_csv(OUTPUT_DIR / "cluster_model_selection.csv", index=False)

    primary_labels = ordered_cluster_labels(kmeans_labels(embeddings, PRIMARY_K))
    assignments, summary, representatives = cluster_summary(
        frame, embeddings, primary_labels, PRIMARY_METHOD, PRIMARY_K
    )
    assignments.to_csv(OUTPUT_DIR / "activity_cluster_assignments_primary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary_primary.csv", index=False)
    representatives.to_csv(OUTPUT_DIR / "cluster_representative_phrases_primary.csv", index=False)

    # Save alternative solutions for audit/sensitivity only.
    agglomerative_rows = []
    kmeans_rows = []
    for k in CANDIDATE_K:
        labels = ordered_cluster_labels(agglomerative_labels(distance, k))
        agglomerative_rows.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"],
                    "ticker": frame["ticker"],
                    "activity_phrase_clean": frame["activity_phrase_clean"],
                    "method": "agglomerative_average_cosine",
                    "k": k,
                    "cluster_id": [f"k{k:02d}_C{label + 1:02d}" for label in labels],
                    "cluster_label_numeric": labels + 1,
                }
            )
        )
        km_labels = ordered_cluster_labels(kmeans_labels(embeddings, k))
        kmeans_rows.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"],
                    "ticker": frame["ticker"],
                    "activity_phrase_clean": frame["activity_phrase_clean"],
                    "method": "kmeans_embedding",
                    "k": k,
                    "cluster_id": [f"k{k:02d}_C{label + 1:02d}" for label in km_labels],
                    "cluster_label_numeric": km_labels + 1,
                }
            )
        )
    pd.concat(agglomerative_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "activity_cluster_assignments_agglomerative_sensitivity.csv",
        index=False,
    )
    pd.concat(kmeans_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "activity_cluster_assignments_kmeans_sensitivity.csv",
        index=False,
    )

    np.save(OUTPUT_DIR / "firm_only_522_mpnet_embeddings_used_for_clustering.npy", embeddings)

    readme = f"""# Step 4 clustering outputs

Input: `{INPUT}`

Rows clustered: {len(frame)}

Embedding model: `{MODEL_NAME}`

Primary clustering method: k-means clustering over normalized MPNet embeddings.

Primary k: {PRIMARY_K}

Important: this folder contains Step 4 clustering only. It does not perform Step 5 theory mapping or taxonomy validation.

Main outputs:

- `activity_cluster_assignments_primary.csv`: one row per activity phrase with its emergent category assignment.
- `cluster_summary_primary.csv`: one row per emergent category with suggested labels, top terms, size, firms, and representative phrases.
- `cluster_representative_phrases_primary.csv`: top phrases closest to each cluster centroid.
- `cluster_model_selection.csv`: metrics across candidate k values.
- `activity_cluster_assignments_agglomerative_sensitivity.csv`: alternative agglomerative assignments for candidate k values.
- `activity_cluster_assignments_kmeans_sensitivity.csv`: alternative k-means assignments for candidate k values.
"""
    (OUTPUT_DIR / "README_step4_clustering.md").write_text(readme)

    print(f"Clustered rows: {len(frame)}")
    print(f"Primary method: {PRIMARY_METHOD}")
    print(f"Primary k: {PRIMARY_K}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Primary cluster size distribution:")
    print(summary[["activity_cluster_id", "cluster_size", "suggested_emergent_category_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
