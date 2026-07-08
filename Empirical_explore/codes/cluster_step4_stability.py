#!/usr/bin/env python3
"""Stability checks for Step 4 clustering results.

Creates /Empirical_explore/stability and compares stability reruns against
original results in /Empirical_explore/clustering using Adjusted Rand Index.

Stability tests:
  1. random-seed reruns;
  2. leave Tesla (TSLA) out;
  3. repeated 80% subsampling.
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
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
INPUT = (
    PROJECT_DIR
    / "phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv"
)
EMBEDDINGS = PROJECT_DIR / "embedding/firm_only_deduplicated_mpnet_embeddings.npy"
ORIGINAL_CLUSTERING = PROJECT_DIR / "clustering"
STABILITY_ROOT = PROJECT_DIR / "stability"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

K_VALUES = [20, 50, 100, 200]
HDBSCAN_PARAMS = [
    {"min_cluster_size": 5, "min_samples": 1},
    {"min_cluster_size": 5, "min_samples": 3},
    {"min_cluster_size": 10, "min_samples": 3},
    {"min_cluster_size": 15, "min_samples": 5},
    {"min_cluster_size": 25, "min_samples": 5},
]
RANDOM_SEEDS = [7, 13, 29]
SUBSAMPLE_SEEDS = [101, 202, 303, 404, 505]
SUBSAMPLE_FRAC = 0.80

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


def folder_for_hdbscan(params: dict[str, int]) -> str:
    return f"min_cluster_size_{params['min_cluster_size']:02d}_min_samples_{params['min_samples']:02d}"


def all_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for params in HDBSCAN_PARAMS:
        runs.append({"method": "hdbscan", "run_name": folder_for_hdbscan(params), "params": params})
    for k in K_VALUES:
        runs.append({"method": "kmeans", "run_name": f"k_{k:03d}", "params": {"k": k}})
    runs.append(
        {
            "method": "bertopic",
            "run_name": "default_min_topic_size_05",
            "params": {
                "min_topic_size": 5,
                "umap_n_neighbors": 15,
                "umap_n_components": 10,
                "hdbscan_min_cluster_size": 5,
                "hdbscan_min_samples": 1,
            },
        }
    )
    for k in K_VALUES:
        runs.append({"method": "hierarchical_ward", "run_name": f"k_{k:03d}", "params": {"k": k}})
    return runs


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
    return ", ".join(names[np.argsort(scores)[::-1][:n]].tolist())


def stable_renumber(labels: np.ndarray, keep_noise: bool = True) -> np.ndarray:
    out = np.full_like(labels, fill_value=-1, dtype=int)
    unique = [x for x in np.unique(labels) if not (keep_noise and x == -1)]
    sizes = pd.Series(labels).value_counts()
    first_row = {label: int(np.where(labels == label)[0][0]) for label in unique}
    ordered = sorted(unique, key=lambda label: (-int(sizes[label]), first_row[label]))
    for new_label, old_label in enumerate(ordered):
        out[labels == old_label] = new_label
    return out


def cluster_id(label: int) -> str:
    return "NOISE" if label == -1 else f"C{label + 1:03d}"


def run_labels(
    method: str,
    run_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    if method == "hdbscan":
        model = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )
        labels = stable_renumber(model.fit_predict(embeddings))
        return labels, pd.DataFrame({"hdbscan_probability": model.probabilities_})

    if method == "kmeans":
        k = min(int(params["k"]), len(frame))
        labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(embeddings)
        return stable_renumber(labels, keep_noise=False), None

    if method == "hierarchical_ward":
        k = min(int(params["k"]), len(frame))
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward", affinity="euclidean")
        labels = model.fit_predict(embeddings)
        return stable_renumber(labels, keep_noise=False), None

    if method == "bertopic":
        docs = frame["activity_phrase_clean"].astype(str).tolist()
        umap_model = UMAP(
            n_neighbors=min(params["umap_n_neighbors"], max(len(frame) - 1, 2)),
            n_components=params["umap_n_components"],
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=params["hdbscan_min_cluster_size"],
            min_samples=params["hdbscan_min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1),
            min_topic_size=params["min_topic_size"],
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
        extra = pd.DataFrame({"bertopic_original_topic": topics})
        return stable_renumber(np.array(topics, dtype=int)), extra

    raise ValueError(f"Unknown method: {method}")


def write_run_outputs(
    test_type: str,
    test_run: str,
    method: str,
    run_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    extra: pd.DataFrame | None,
    seed: int,
) -> Path:
    run_dir = STABILITY_ROOT / test_type / test_run / method / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    assignments = frame.copy()
    assignments.insert(0, "cluster_id", [cluster_id(int(x)) for x in labels])
    assignments.insert(1, "cluster_label_numeric", labels.astype(int))
    assignments.insert(2, "clustering_method", method)
    assignments.insert(3, "clustering_run", run_name)
    assignments.insert(4, "stability_test_type", test_type)
    assignments.insert(5, "stability_test_run", test_run)
    assignments["random_seed"] = seed
    assignments["embedding_model"] = MODEL_NAME
    if extra is not None:
        for col in extra.columns:
            assignments[col] = extra[col].values
    assignments.to_csv(run_dir / "assignments.csv", index=False)

    summary_rows = []
    rep_rows = []
    for label in sorted(np.unique(labels), key=lambda x: (x == -1, x)):
        idx = np.where(labels == label)[0]
        sub = frame.iloc[idx]
        sub_emb = embeddings[idx]
        phrases = sub["activity_phrase_clean"].astype(str).tolist()
        centroid = sub_emb.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = sub_emb @ centroid
        order = np.argsort(sims)[::-1]
        pairwise = cosine_similarity(sub_emb) if len(sub_emb) > 1 else np.array([[1.0]])
        within = pairwise[np.triu_indices_from(pairwise, k=1)] if len(sub_emb) > 1 else np.array([])
        cid = cluster_id(int(label))
        summary_rows.append(
            {
                "cluster_id": cid,
                "cluster_label_numeric": int(label),
                "is_noise": bool(label == -1),
                "top_terms": "noise / unclustered" if label == -1 else top_terms(phrases, 10),
                "cluster_size": int(len(sub)),
                "n_firms": int(sub["ticker"].nunique()),
                "firms": "|".join(sorted(sub["ticker"].astype(str).unique())),
                "mean_within_cluster_pairwise_cosine": float(np.mean(within)) if len(within) else np.nan,
                "representative_phrase_1": sub.iloc[int(order[0])]["activity_phrase_clean"],
            }
        )
        for rank, local_pos in enumerate(order[:10], start=1):
            row = sub.iloc[int(local_pos)]
            rep_rows.append(
                {
                    "cluster_id": cid,
                    "rank": rank,
                    "similarity_to_centroid": float(sims[int(local_pos)]),
                    "row_id": row["row_id"],
                    "ticker": row["ticker"],
                    "activity_phrase_clean": row["activity_phrase_clean"],
                    "source_sentence": row["source_sentence"],
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "cluster_summary.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(run_dir / "representative_phrases.csv", index=False)

    metadata = {
        "test_type": test_type,
        "test_run": test_run,
        "method": method,
        "run_name": run_name,
        "params": params,
        "random_seed": seed,
        "n_rows": int(len(frame)),
        "n_clusters_excluding_noise": int((summary["is_noise"] == False).sum()),
        "n_noise_rows": int((labels == -1).sum()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def original_assignments(method: str, run_name: str) -> pd.DataFrame:
    path = ORIGINAL_CLUSTERING / method / run_name / "assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[["row_id", "cluster_label_numeric", "cluster_id"]].rename(
        columns={
            "cluster_label_numeric": "original_cluster_label_numeric",
            "cluster_id": "original_cluster_id",
        }
    )


def ari_against_original(method: str, run_name: str, stability_assignments: pd.DataFrame) -> dict[str, Any]:
    original = original_assignments(method, run_name)
    merged = original.merge(
        stability_assignments[["row_id", "cluster_label_numeric", "cluster_id"]],
        on="row_id",
        how="inner",
    )
    if len(merged) < 2:
        score = np.nan
    else:
        score = adjusted_rand_score(
            merged["original_cluster_label_numeric"].astype(int),
            merged["cluster_label_numeric"].astype(int),
        )
    return {
        "method": method,
        "run_name": run_name,
        "n_compared_rows": int(len(merged)),
        "ari_vs_original": float(score) if not pd.isna(score) else np.nan,
        "original_n_clusters_compared": int(merged["original_cluster_label_numeric"].nunique()),
        "stability_n_clusters_compared": int(merged["cluster_label_numeric"].nunique()),
    }


def run_stability_case(
    test_type: str,
    test_run: str,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    ari_rows = []
    for run in all_runs():
        method, run_name, params = run["method"], run["run_name"], run["params"]
        labels, extra = run_labels(method, run_name, params, frame, embeddings, seed)
        write_run_outputs(test_type, test_run, method, run_name, params, frame, embeddings, labels, extra, seed)
        assignments = pd.DataFrame(
            {
                "row_id": frame["row_id"].values,
                "cluster_label_numeric": labels.astype(int),
                "cluster_id": [cluster_id(int(x)) for x in labels],
            }
        )
        row = ari_against_original(method, run_name, assignments)
        row.update(
            {
                "test_type": test_type,
                "test_run": test_run,
                "random_seed": seed,
                "n_rows_in_stability_run": int(len(frame)),
                "n_noise_rows_stability": int((labels == -1).sum()),
            }
        )
        ari_rows.append(row)
    return ari_rows


def main() -> None:
    STABILITY_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    embeddings = np.load(EMBEDDINGS).astype(np.float32)
    if len(frame) != embeddings.shape[0]:
        raise ValueError("Input and embedding row counts do not match")
    frame["row_id"] = frame["row_id"].astype(int)
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms[:, None], 1e-12)

    all_ari = []

    # 1. Random seed reruns. HDBSCAN and Ward are deterministic, but included as
    # comparable reruns for the full method grid.
    for seed in RANDOM_SEEDS:
        all_ari.extend(
            run_stability_case(
                "random_seed",
                f"seed_{seed}",
                frame.copy(),
                embeddings.copy(),
                seed,
            )
        )

    # 2. Leave Tesla out.
    no_tesla_mask = frame["ticker"].astype(str).ne("TSLA").to_numpy()
    all_ari.extend(
        run_stability_case(
            "leave_tesla_out",
            "exclude_TSLA",
            frame.loc[no_tesla_mask].reset_index(drop=True).copy(),
            embeddings[no_tesla_mask].copy(),
            42,
        )
    )

    # 3. 80% subsampling repeated several times.
    n = len(frame)
    sample_n = int(round(SUBSAMPLE_FRAC * n))
    for seed in SUBSAMPLE_SEEDS:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(np.arange(n), size=sample_n, replace=False))
        all_ari.extend(
            run_stability_case(
                "subsample_80pct",
                f"seed_{seed}",
                frame.iloc[sample_idx].reset_index(drop=True).copy(),
                embeddings[sample_idx].copy(),
                seed,
            )
        )

    ari = pd.DataFrame(all_ari)
    ari.to_csv(STABILITY_ROOT / "ari_against_original_all_runs.csv", index=False)

    summary = (
        ari.groupby(["test_type", "method", "run_name"], dropna=False)
        .agg(
            n_tests=("ari_vs_original", "size"),
            mean_ari=("ari_vs_original", "mean"),
            min_ari=("ari_vs_original", "min"),
            max_ari=("ari_vs_original", "max"),
            mean_compared_rows=("n_compared_rows", "mean"),
        )
        .reset_index()
        .sort_values(["test_type", "method", "run_name"])
    )
    summary.to_csv(STABILITY_ROOT / "ari_summary_by_test_method_run.csv", index=False)

    method_summary = (
        ari.groupby(["test_type", "method"], dropna=False)
        .agg(
            n_tests=("ari_vs_original", "size"),
            mean_ari=("ari_vs_original", "mean"),
            median_ari=("ari_vs_original", "median"),
            min_ari=("ari_vs_original", "min"),
            max_ari=("ari_vs_original", "max"),
        )
        .reset_index()
        .sort_values(["test_type", "method"])
    )
    method_summary.to_csv(STABILITY_ROOT / "ari_summary_by_test_method.csv", index=False)

    readme = f"""# Clustering stability checks

Baseline clustering folder: `{ORIGINAL_CLUSTERING}`

Input phrases: `{INPUT}`

Embeddings: `{EMBEDDINGS}`

Stability checks:

1. `random_seed/`: reruns with seeds {RANDOM_SEEDS}. K-means and BERTopic are seed-sensitive; HDBSCAN and Ward are deterministic but rerun for comparison.
2. `leave_tesla_out/`: excludes TSLA and reclusters the other three firms.
3. `subsample_80pct/`: reclusters {int(SUBSAMPLE_FRAC * 100)}% phrase samples for seeds {SUBSAMPLE_SEEDS}.

ARI outputs:

- `ari_against_original_all_runs.csv`: one row per stability run vs original run.
- `ari_summary_by_test_method_run.csv`: ARI summarized by test/method/run.
- `ari_summary_by_test_method.csv`: compact method-level summary.
"""
    (STABILITY_ROOT / "README_stability.md").write_text(readme)

    print("ARI method-level summary:")
    print(method_summary.to_string(index=False))
    print(f"Saved stability outputs to: {STABILITY_ROOT}")


if __name__ == "__main__":
    main()
