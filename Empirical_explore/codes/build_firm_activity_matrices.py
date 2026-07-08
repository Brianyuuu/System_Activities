#!/usr/bin/env python3
"""Step 6: build firm-year × activity-cluster matrices.

For every clustering run under /Empirical_explore/clustering, create:
  - binary firm-year × cluster matrix
  - count firm-year × cluster matrix
  - weighted share matrix: count / total non-noise activity mentions for firm-year
  - weighted TF-IDF matrix: count * log((1+n_firmyears)/(1+df_cluster)) + 1

Noise/unclustered rows are excluded from the wide matrices because they are not
activity clusters, but they are retained in the long audit output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
CLUSTERING_DIR = PROJECT_DIR / "clustering"
OUTPUT_ROOT = PROJECT_DIR / "firm_act_matrix"

ID_COLS = ["firm_id", "ticker", "company", "year"]


def discover_assignment_files() -> list[Path]:
    files = []
    for path in CLUSTERING_DIR.rglob("assignments.csv"):
        rel = path.relative_to(CLUSTERING_DIR)
        if len(rel.parts) >= 3:
            files.append(path)
    return sorted(files)


def run_parts(path: Path) -> tuple[str, str]:
    rel = path.relative_to(CLUSTERING_DIR)
    return rel.parts[0], rel.parts[1]


def run_output_dir(method: str, run_name: str) -> Path:
    return OUTPUT_ROOT / method / run_name


def is_noise_cluster(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("NOISE")


def matrix_from_counts(counts: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = (
        counts.pivot_table(
            index=ID_COLS,
            columns="cluster_id",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )
    matrix.columns.name = None
    cluster_cols = sorted([c for c in matrix.columns if c not in ID_COLS])
    return matrix[ID_COLS + cluster_cols]


def add_missing_firm_years(matrix: pd.DataFrame, firm_years: pd.DataFrame, fill_value: float = 0) -> pd.DataFrame:
    out = firm_years.merge(matrix, on=ID_COLS, how="left")
    cluster_cols = [c for c in out.columns if c not in ID_COLS]
    out[cluster_cols] = out[cluster_cols].fillna(fill_value)
    return out


def build_for_run(path: Path) -> dict[str, Any]:
    method, run_name = run_parts(path)
    out_dir = run_output_dir(method, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    missing = [c for c in ID_COLS + ["cluster_id", "row_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["year"] = df["year"].astype(str)
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["is_noise_cluster"] = is_noise_cluster(df["cluster_id"])

    firm_years = df[ID_COLS].drop_duplicates().sort_values(["ticker", "year"]).reset_index(drop=True)
    all_long = (
        df.groupby(ID_COLS + ["cluster_id", "is_noise_cluster"], dropna=False)
        .agg(
            activity_phrase_count=("row_id", "count"),
            row_ids=("row_id", lambda x: "|".join(x.astype(str))),
            example_phrases=("activity_phrase_clean", lambda x: " || ".join(x.astype(str).head(10))),
        )
        .reset_index()
    )
    all_long.to_csv(out_dir / "firm_year_cluster_counts_long_with_noise.csv", index=False)

    non_noise = all_long[~all_long["is_noise_cluster"]].copy()
    if non_noise.empty:
        # Create empty matrices with firm-year identifiers if a run clusters
        # nothing but noise.
        for name in [
            "binary_matrix.csv",
            "count_matrix.csv",
            "weighted_share_matrix.csv",
            "weighted_tfidf_matrix.csv",
        ]:
            firm_years.to_csv(out_dir / name, index=False)
        metadata = {
            "method": method,
            "run_name": run_name,
            "source_assignments": str(path),
            "n_firm_years": int(len(firm_years)),
            "n_activity_rows": int(len(df)),
            "n_non_noise_activity_rows": 0,
            "n_noise_activity_rows": int(df["is_noise_cluster"].sum()),
            "n_activity_clusters": 0,
        }
        (out_dir / "matrix_metadata.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    non_noise["binary_value"] = 1
    count_matrix = add_missing_firm_years(
        matrix_from_counts(non_noise, "activity_phrase_count"), firm_years, 0
    )
    binary_matrix = count_matrix.copy()
    cluster_cols = [c for c in binary_matrix.columns if c not in ID_COLS]
    binary_matrix[cluster_cols] = (binary_matrix[cluster_cols] > 0).astype(int)

    share_matrix = count_matrix.copy()
    totals = count_matrix[cluster_cols].sum(axis=1).replace(0, np.nan)
    share_matrix[cluster_cols] = share_matrix[cluster_cols].div(totals, axis=0).fillna(0)

    tfidf_matrix = count_matrix.copy()
    n_firm_years = len(count_matrix)
    df_cluster = (count_matrix[cluster_cols] > 0).sum(axis=0)
    idf = np.log((1 + n_firm_years) / (1 + df_cluster)) + 1
    tfidf_matrix[cluster_cols] = tfidf_matrix[cluster_cols] * idf

    binary_matrix.to_csv(out_dir / "binary_matrix.csv", index=False)
    count_matrix.to_csv(out_dir / "count_matrix.csv", index=False)
    share_matrix.to_csv(out_dir / "weighted_share_matrix.csv", index=False)
    tfidf_matrix.to_csv(out_dir / "weighted_tfidf_matrix.csv", index=False)

    dictionary = pd.DataFrame(
        {
            "cluster_id": cluster_cols,
            "document_frequency_firm_years": [int(df_cluster[c]) for c in cluster_cols],
            "idf": [float(idf[c]) for c in cluster_cols],
        }
    )
    cluster_summary_path = path.parent / "cluster_summary.csv"
    if cluster_summary_path.exists():
        cluster_summary = pd.read_csv(cluster_summary_path)
        if "cluster_id" in cluster_summary.columns:
            dictionary = dictionary.merge(
                cluster_summary,
                on="cluster_id",
                how="left",
                suffixes=("", "_cluster_summary"),
            )
    dictionary.to_csv(out_dir / "cluster_dictionary.csv", index=False)

    noise_rows = int(df["is_noise_cluster"].sum())
    metadata = {
        "method": method,
        "run_name": run_name,
        "source_assignments": str(path),
        "n_firm_years": int(len(firm_years)),
        "n_activity_rows": int(len(df)),
        "n_non_noise_activity_rows": int((~df["is_noise_cluster"]).sum()),
        "n_noise_activity_rows": noise_rows,
        "n_activity_clusters": int(len(cluster_cols)),
        "matrix_rows": int(len(count_matrix)),
        "matrix_cluster_columns": int(len(cluster_cols)),
        "weighted_share_definition": "count_ij / sum_j count_ij over non-noise clusters for firm-year i",
        "weighted_tfidf_definition": "count_ij * (log((1+n_firmyears)/(1+df_j)) + 1)",
    }
    (out_dir / "matrix_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assignment_files = discover_assignment_files()
    if not assignment_files:
        raise FileNotFoundError(f"No assignments.csv files found under {CLUSTERING_DIR}")

    metadata_rows = []
    for path in assignment_files:
        metadata_rows.append(build_for_run(path))

    summary = pd.DataFrame(metadata_rows).sort_values(["method", "run_name"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_ROOT / "firm_activity_matrix_run_summary.csv", index=False)

    readme = f"""# Step 6 firm-year × activity-cluster matrices

Input clustering folder: `{CLUSTERING_DIR}`

Output folder: `{OUTPUT_ROOT}`

For every clustering run, this folder contains four wide firm-year × cluster matrices:

- `binary_matrix.csv`: 1 if the firm-year mentions at least one activity in the cluster, otherwise 0.
- `count_matrix.csv`: number of activity phrases from the firm-year assigned to the cluster.
- `weighted_share_matrix.csv`: count normalized by the firm-year's total non-noise activity mentions.
- `weighted_tfidf_matrix.csv`: count multiplied by `log((1+n_firmyears)/(1+df_cluster)) + 1`.

Each run folder also contains:

- `firm_year_cluster_counts_long_with_noise.csv`: long audit file, including `NOISE` when present.
- `cluster_dictionary.csv`: cluster columns, document frequency, IDF, and cluster summary metadata when available.
- `matrix_metadata.json`: run-level matrix metadata.

`NOISE`/unclustered labels are excluded from the wide matrices because they are not activity clusters, but they are retained in the long audit file.
"""
    (OUTPUT_ROOT / "README_firm_activity_matrices.md").write_text(readme)

    print(f"Processed clustering runs: {len(summary)}")
    print(summary[["method", "run_name", "n_firm_years", "n_activity_clusters", "n_noise_activity_rows"]].to_string(index=False))
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
