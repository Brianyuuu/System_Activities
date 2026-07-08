#!/usr/bin/env python3
"""Step 4 clustering grid over several methods and granularities.

Inputs:
  - final 522 firm-only deduplicated activity phrases
  - MPNet embeddings created earlier

Outputs:
  /phrases/clean_normalize/clustering/
    hdbscan/min_cluster_size_XX_min_samples_YY/
    kmeans/k_XXX/
    bertopic/default_min_topic_size_05/
    hierarchical_ward/k_XXX/

Each run folder contains assignments.csv, cluster_summary.csv,
representative_phrases.csv, and run_metadata.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
INPUT = (
    PROJECT_DIR
    / "phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv"
)
EMBEDDINGS = (
    PROJECT_DIR
    / "phrases/clean_normalize/embedding/firm_only_deduplicated_mpnet_embeddings.npy"
)
OUTPUT_ROOT = PROJECT_DIR / "phrases/clean_normalize/clustering"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

K_VALUES = [20, 50, 100, 200]
HDBSCAN_PARAMS = [
    {"min_cluster_size": 5, "min_samples": 1},
    {"min_cluster_size": 5, "min_samples": 3},
    {"min_cluster_size": 10, "min_samples": 3},
    {"min_cluster_size": 15, "min_samples": 5},
    {"min_cluster_size": 25, "min_samples": 5},
]

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


def clean_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file in ["assignments.csv", "cluster_summary.csv", "representative_phrases.csv", "run_metadata.json"]:
        target = path / file
        if target.exists():
            target.unlink()


def top_terms(phrases: list[str], n: int = 10) -> str:
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
    return ", ".join(names[np.argsort(scores)[::-1][:n]].tolist())


def stable_renumber(labels: np.ndarray, keep_noise: bool = True) -> np.ndarray:
    """Renumber cluster labels by descending size while preserving -1 noise."""

    out = np.full_like(labels, fill_value=-1, dtype=int)
    unique = [x for x in np.unique(labels) if not (keep_noise and x == -1)]
    sizes = pd.Series(labels).value_counts()
    first_row = {label: int(np.where(labels == label)[0][0]) for label in unique}
    ordered = sorted(unique, key=lambda label: (-int(sizes[label]), first_row[label]))
    for new_label, old_label in enumerate(ordered):
        out[labels == old_label] = new_label
    return out


def cluster_id_for_label(label: int) -> str:
    return "NOISE" if label == -1 else f"C{label + 1:03d}"


def metric_summary(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    non_noise = labels != -1
    eval_labels = labels[non_noise]
    eval_embeddings = embeddings[non_noise]
    n_clusters = len([x for x in np.unique(labels) if x != -1])
    out: dict[str, Any] = {
        "n_rows": int(len(labels)),
        "n_clusters_excluding_noise": int(n_clusters),
        "n_noise_rows": int((labels == -1).sum()),
    }
    if n_clusters >= 2 and len(eval_embeddings) > n_clusters:
        out["silhouette_cosine_excluding_noise"] = float(
            silhouette_score(eval_embeddings, eval_labels, metric="cosine")
        )
        out["calinski_harabasz_excluding_noise"] = float(
            calinski_harabasz_score(eval_embeddings, eval_labels)
        )
        out["davies_bouldin_excluding_noise"] = float(
            davies_bouldin_score(eval_embeddings, eval_labels)
        )
    else:
        out["silhouette_cosine_excluding_noise"] = None
        out["calinski_harabasz_excluding_noise"] = None
        out["davies_bouldin_excluding_noise"] = None
    return out


def write_outputs(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str,
    run_name: str,
    run_dir: Path,
    params: dict[str, Any],
    extra_assignments: pd.DataFrame | None = None,
) -> dict[str, Any]:
    clean_folder(run_dir)
    labels = labels.astype(int)
    cluster_ids = [cluster_id_for_label(label) for label in labels]

    assignments = frame.copy()
    assignments.insert(0, "cluster_id", cluster_ids)
    assignments.insert(1, "cluster_label_numeric", labels)
    assignments.insert(2, "clustering_method", method)
    assignments.insert(3, "clustering_run", run_name)
    assignments["embedding_model"] = MODEL_NAME
    if extra_assignments is not None:
        for col in extra_assignments.columns:
            assignments[col] = extra_assignments[col].values
    assignments.to_csv(run_dir / "assignments.csv", index=False)

    summary_rows = []
    rep_rows = []
    for label in sorted(np.unique(labels), key=lambda x: (x == -1, x)):
        idx = np.where(labels == label)[0]
        sub = frame.iloc[idx]
        sub_embeddings = embeddings[idx]
        phrases = sub["activity_phrase_clean"].astype(str).tolist()
        if label == -1:
            centroid = sub_embeddings.mean(axis=0)
        else:
            centroid = sub_embeddings.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = sub_embeddings @ centroid
        order = np.argsort(sims)[::-1]
        pairwise = cosine_similarity(sub_embeddings) if len(sub_embeddings) > 1 else np.array([[1.0]])
        within = pairwise[np.triu_indices_from(pairwise, k=1)] if len(sub_embeddings) > 1 else np.array([])
        cid = cluster_id_for_label(int(label))
        summary_rows.append(
            {
                "cluster_id": cid,
                "cluster_label_numeric": int(label),
                "is_noise": bool(label == -1),
                "suggested_emergent_category_label": "noise / unclustered"
                if label == -1
                else top_terms(phrases, 4),
                "top_terms": top_terms(phrases, 10),
                "cluster_size": int(len(sub)),
                "n_firms": int(sub["ticker"].nunique()),
                "firms": "|".join(sorted(sub["ticker"].astype(str).unique())),
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
                    "cluster_id": cid,
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

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "cluster_summary.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(run_dir / "representative_phrases.csv", index=False)

    metrics = metric_summary(embeddings, labels)
    metadata = {
        "method": method,
        "run_name": run_name,
        "params": params,
        "input": str(INPUT),
        "n_input_rows": int(len(frame)),
        "embedding_model": MODEL_NAME,
        **metrics,
        "min_cluster_size_observed_excluding_noise": int(
            summary.loc[~summary["is_noise"], "cluster_size"].min()
        )
        if (~summary["is_noise"]).any()
        else 0,
        "max_cluster_size_observed_excluding_noise": int(
            summary.loc[~summary["is_noise"], "cluster_size"].max()
        )
        if (~summary["is_noise"]).any()
        else 0,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def run_hdbscan(frame: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for params in HDBSCAN_PARAMS:
        run_name = (
            f"min_cluster_size_{params['min_cluster_size']:02d}_"
            f"min_samples_{params['min_samples']:02d}"
        )
        model = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )
        labels = stable_renumber(model.fit_predict(embeddings))
        probabilities = pd.DataFrame({"hdbscan_probability": model.probabilities_})
        rows.append(
            write_outputs(
                frame,
                embeddings,
                labels,
                "hdbscan",
                run_name,
                OUTPUT_ROOT / "hdbscan" / run_name,
                params,
                extra_assignments=probabilities,
            )
        )
    return rows


def run_kmeans(frame: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for k in K_VALUES:
        run_name = f"k_{k:03d}"
        labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(embeddings)
        labels = stable_renumber(labels, keep_noise=False)
        rows.append(
            write_outputs(
                frame,
                embeddings,
                labels,
                "kmeans",
                run_name,
                OUTPUT_ROOT / "kmeans" / run_name,
                {"k": k, "n_init": 50, "random_state": 42},
            )
        )
    return rows


def run_hierarchical_ward(frame: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    linkage = sch.linkage(embeddings, method="ward", metric="euclidean")
    ward_root = OUTPUT_ROOT / "hierarchical_ward"
    ward_root.mkdir(parents=True, exist_ok=True)
    np.save(ward_root / "ward_linkage_matrix.npy", linkage)
    pd.DataFrame(linkage, columns=["cluster_1", "cluster_2", "distance", "new_cluster_size"]).to_csv(
        ward_root / "ward_linkage_matrix.csv", index=False
    )
    for k in K_VALUES:
        run_name = f"k_{k:03d}"
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward", affinity="euclidean")
        labels = stable_renumber(model.fit_predict(embeddings), keep_noise=False)
        rows.append(
            write_outputs(
                frame,
                embeddings,
                labels,
                "hierarchical_ward",
                run_name,
                OUTPUT_ROOT / "hierarchical_ward" / run_name,
                {"k": k, "linkage": "ward", "metric": "euclidean"},
            )
        )
    return rows


def run_bertopic(frame: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    docs = frame["activity_phrase_clean"].astype(str).tolist()
    run_name = "default_min_topic_size_05"
    params = {
        "min_topic_size": 5,
        "umap_n_neighbors": 15,
        "umap_n_components": 10,
        "hdbscan_min_cluster_size": 5,
        "hdbscan_min_samples": 1,
    }
    umap_model = UMAP(
        n_neighbors=params["umap_n_neighbors"],
        n_components=params["umap_n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=params["hdbscan_min_cluster_size"],
        min_samples=params["hdbscan_min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1)
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=params["min_topic_size"],
        calculate_probabilities=False,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    labels = stable_renumber(np.array(topics, dtype=int))

    topic_info = topic_model.get_topic_info()
    run_dir = OUTPUT_ROOT / "bertopic" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    topic_info.to_csv(run_dir / "bertopic_topic_info.csv", index=False)

    topic_word_rows = []
    for topic in sorted(set(topics)):
        for rank, (word, score) in enumerate(topic_model.get_topic(topic) or [], start=1):
            topic_word_rows.append({"bertopic_topic": topic, "rank": rank, "word": word, "score": score})
    pd.DataFrame(topic_word_rows).to_csv(run_dir / "bertopic_topic_words.csv", index=False)

    extra = pd.DataFrame({"bertopic_original_topic": topics})
    metadata = write_outputs(
        frame,
        embeddings,
        labels,
        "bertopic",
        run_name,
        run_dir,
        params,
        extra_assignments=extra,
    )
    return [metadata]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    embeddings = np.load(EMBEDDINGS).astype(np.float32)
    if len(frame) != 522:
        raise ValueError(f"Expected 522 input rows, found {len(frame)}")
    if embeddings.shape[0] != len(frame):
        raise ValueError(f"Embedding rows {embeddings.shape[0]} do not match input rows {len(frame)}")
    if not frame["actor_class"].eq("firm").all():
        raise ValueError("Input contains non-firm actor rows")
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms[:, None], 1e-12)

    all_metadata = []
    all_metadata.extend(run_hdbscan(frame, embeddings))
    all_metadata.extend(run_kmeans(frame, embeddings))
    all_metadata.extend(run_bertopic(frame, embeddings))
    all_metadata.extend(run_hierarchical_ward(frame, embeddings))

    summary = pd.DataFrame(all_metadata)
    summary.to_csv(OUTPUT_ROOT / "clustering_grid_run_summary.csv", index=False)

    readme = f"""# Step 4 clustering method grid

Input: `{INPUT}`

Rows clustered: {len(frame)}

Embedding model: `{MODEL_NAME}`

This folder contains Step 4 clustering only. It does not perform Step 5 theory mapping.

Folder layout:

- `hdbscan/min_cluster_size_XX_min_samples_YY/`
- `kmeans/k_020/`, `kmeans/k_050/`, `kmeans/k_100/`, `kmeans/k_200/`
- `bertopic/default_min_topic_size_05/`
- `hierarchical_ward/k_020/`, `hierarchical_ward/k_050/`, `hierarchical_ward/k_100/`, `hierarchical_ward/k_200/`

Each run folder contains:

- `assignments.csv`: one row per activity phrase with cluster assignment
- `cluster_summary.csv`: one row per cluster/topic
- `representative_phrases.csv`: up to 10 central phrases per cluster/topic
- `run_metadata.json`: parameters and compact metrics

The file `clustering_grid_run_summary.csv` summarizes all runs.
"""
    (OUTPUT_ROOT / "README_clustering_grid.md").write_text(readme)
    print(summary[["method", "run_name", "n_clusters_excluding_noise", "n_noise_rows", "silhouette_cosine_excluding_noise", "min_cluster_size_observed_excluding_noise", "max_cluster_size_observed_excluding_noise"]].to_string(index=False))
    print(f"Saved clustering grid to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
