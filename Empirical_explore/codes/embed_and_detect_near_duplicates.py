#!/usr/bin/env python3
"""Embed final firm-only activities and detect near-duplicate phrase pairs.

The script uses sentence-transformers/all-mpnet-base-v2 on the final
firm-only deduplicated activity file, computes cosine similarities among all
retained phrases, and reports near-duplicate pairs/groups for thresholds from
0.88 to 0.95.

Each threshold-specific group CSV is expanded: one duplicate-group member
phrase per row, with group metadata repeated across member rows.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
INPUT = (
    PROJECT_DIR
    / "phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv"
)
OUTPUT_DIR = PROJECT_DIR / "phrases/clean_normalize/embedding"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
THRESHOLDS = [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
PAIR_OUTPUT_NAME = "near_duplicate_pairs_threshold_088_to_095.csv"
SUMMARY_OUTPUT_NAME = "near_duplicate_threshold_summary_088_to_095.csv"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def threshold_name(threshold: float) -> str:
    return f"{int(round(threshold * 100)):02d}"


def build_pairs(frame: pd.DataFrame, sims: np.ndarray, min_threshold: float) -> pd.DataFrame:
    rows = []
    n = len(frame)
    upper_i, upper_j = np.triu_indices(n, k=1)
    mask = sims[upper_i, upper_j] >= min_threshold

    for i, j, sim in zip(upper_i[mask], upper_j[mask], sims[upper_i[mask], upper_j[mask]]):
        left = frame.iloc[int(i)]
        right = frame.iloc[int(j)]
        rows.append(
            {
                "left_embedding_row": int(i),
                "right_embedding_row": int(j),
                "left_row_id": left["row_id"],
                "right_row_id": right["row_id"],
                "left_ticker": left["ticker"],
                "right_ticker": right["ticker"],
                "same_ticker": left["ticker"] == right["ticker"],
                "same_year": left["year"] == right["year"],
                "cosine_similarity": float(sim),
                "left_activity_phrase_clean": left["activity_phrase_clean"],
                "right_activity_phrase_clean": right["activity_phrase_clean"],
                "left_activity_phrase": left["activity_phrase"],
                "right_activity_phrase": right["activity_phrase"],
                "left_source_sentence": left["source_sentence"],
                "right_source_sentence": right["source_sentence"],
                "left_source_method": left["source_method"],
                "right_source_method": right["source_method"],
            }
        )

    pairs = pd.DataFrame(rows).sort_values(
        ["cosine_similarity", "same_ticker", "left_ticker", "right_ticker"],
        ascending=[False, False, True, True],
    )
    for threshold in THRESHOLDS:
        pairs[f"above_{threshold_name(threshold)}"] = pairs["cosine_similarity"] >= threshold
    return pairs


def build_groups(frame: pd.DataFrame, pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    uf = UnionFind(len(frame))
    threshold_pairs = pairs[pairs["cosine_similarity"] >= threshold]
    for _, row in threshold_pairs.iterrows():
        uf.union(int(row["left_embedding_row"]), int(row["right_embedding_row"]))

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(frame)):
        components[uf.find(i)].append(i)

    grouped_indices = split_components_complete_link(list(components.values()), threshold)

    rows = []
    group_num = 1
    for indices in sorted(grouped_indices, key=lambda xs: (len(xs), -min(xs)), reverse=True):
        if len(indices) == 1:
            continue
        group_id = f"ED{threshold_name(threshold)}_{group_num:04d}"
        group_num += 1
        sub = frame.iloc[indices]
        group_pair_sims = []
        for idx, i in enumerate(indices):
            for j in indices[idx + 1 :]:
                group_pair_sims.append(float(sims_global[i, j]))
        max_similarity_to_other = {
            i: max(float(sims_global[i, j]) for j in indices if j != i) for i in indices
        }
        group_meta = {
            "threshold": threshold,
            "embedding_duplicate_group_id": group_id,
            "group_size": len(indices),
            "min_pairwise_cosine_in_group": min(group_pair_sims),
            "max_pairwise_cosine_in_group": max(group_pair_sims),
            "member_embedding_rows": "|".join(str(i) for i in indices),
            "member_row_ids": "|".join(sub["row_id"].astype(str).tolist()),
            "member_tickers": "|".join(sub["ticker"].astype(str).tolist()),
            "unique_tickers": "|".join(sorted(sub["ticker"].astype(str).unique())),
            "all_same_ticker": sub["ticker"].nunique(dropna=False) == 1,
        }
        for i in indices:
            row = frame.iloc[i]
            rows.append(
                {
                    **group_meta,
                    "embedding_row": i,
                    "row_id": row["row_id"],
                    "ticker": row["ticker"],
                    "company": row["company"],
                    "year": row["year"],
                    "activity_phrase_clean": row["activity_phrase_clean"],
                    "activity_phrase": row["activity_phrase"],
                    "source_sentence": row["source_sentence"],
                    "source_method": row["source_method"],
                    "actor": row["actor"],
                    "actor_class": row["actor_class"],
                    "max_similarity_to_other_group_member": max_similarity_to_other[i],
                }
            )
    return pd.DataFrame(rows)


def split_components_complete_link(components: list[list[int]], threshold: float) -> list[list[int]]:
    """Split connected components into disjoint complete-link clusters.

    A connected component can contain weakly related endpoints when similarity
    links form a chain: A-B >= threshold and B-C >= threshold, but A-C <
    threshold. For duplicate review sheets, that makes broad groups difficult
    to audit. This splitter keeps the output disjoint while requiring every
    pair inside a returned multi-row group to meet the threshold.
    """

    split_groups: list[list[int]] = []
    for component in components:
        if len(component) <= 2:
            split_groups.append(component)
            continue

        remaining = set(component)
        while remaining:
            # Start from the remaining node with the strongest average
            # similarity to other remaining nodes; this tends to produce stable,
            # interpretable clusters.
            seed = max(
                remaining,
                key=lambda i: (
                    sum(float(sims_global[i, j]) for j in remaining if j != i)
                    / max(len(remaining) - 1, 1),
                    i,
                ),
            )
            cluster = [seed]
            remaining.remove(seed)

            changed = True
            while changed:
                changed = False
                candidates = sorted(
                    remaining,
                    key=lambda i: (
                        min(float(sims_global[i, j]) for j in cluster),
                        sum(float(sims_global[i, j]) for j in cluster) / len(cluster),
                        i,
                    ),
                    reverse=True,
                )
                for candidate in candidates:
                    if all(float(sims_global[candidate, j]) >= threshold for j in cluster):
                        cluster.append(candidate)
                        remaining.remove(candidate)
                        changed = True
                        break

            split_groups.append(sorted(cluster))

    return split_groups


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(INPUT).reset_index(drop=True)
    if frame["activity_phrase_clean"].isna().any():
        raise ValueError("Missing activity_phrase_clean values in input")
    if not frame["actor_class"].eq("firm").all():
        raise ValueError("Input contains non-firm actor rows")

    model = SentenceTransformer(MODEL_NAME)
    phrases = frame["activity_phrase_clean"].astype(str).tolist()
    embeddings = model.encode(
        phrases,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if embeddings.shape[0] != len(frame):
        raise RuntimeError("Embedding count does not match input row count")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("Embeddings are not unit-normalized")

    global sims_global
    sims_global = embeddings @ embeddings.T

    np.save(OUTPUT_DIR / "firm_only_deduplicated_mpnet_embeddings.npy", embeddings)

    index_cols = [
        "row_id",
        "ticker",
        "company",
        "year",
        "activity_phrase",
        "activity_phrase_clean",
        "source_sentence",
        "source_method",
        "actor",
        "actor_class",
    ]
    index = frame[[c for c in index_cols if c in frame.columns]].copy()
    index.insert(0, "embedding_row", np.arange(len(index), dtype=int))
    index["embedding_model"] = MODEL_NAME
    index.to_csv(OUTPUT_DIR / "firm_only_deduplicated_mpnet_index.csv", index=False)

    pairs = build_pairs(frame, sims_global, min(THRESHOLDS))
    pairs.to_csv(OUTPUT_DIR / PAIR_OUTPUT_NAME, index=False)

    summary_rows = []
    for threshold in THRESHOLDS:
        threshold_pairs = pairs[pairs["cosine_similarity"] >= threshold]
        groups = build_groups(frame, pairs, threshold)
        groups.to_csv(
            OUTPUT_DIR / f"near_duplicate_groups_threshold_{threshold_name(threshold)}.csv",
            index=False,
        )
        summary_rows.append(
            {
                "threshold": threshold,
                "near_duplicate_pairs_all": len(threshold_pairs),
                "near_duplicate_pairs_same_ticker": int(threshold_pairs["same_ticker"].sum()),
                "near_duplicate_pairs_cross_ticker": int((~threshold_pairs["same_ticker"]).sum()),
                "near_duplicate_groups_all": int(groups["embedding_duplicate_group_id"].nunique())
                if len(groups)
                else 0,
                "near_duplicate_groups_same_ticker_only": int(
                    groups.drop_duplicates("embedding_duplicate_group_id")["all_same_ticker"].sum()
                )
                if len(groups)
                else 0,
                "near_duplicate_groups_cross_ticker": int(
                    (~groups.drop_duplicates("embedding_duplicate_group_id")["all_same_ticker"]).sum()
                )
                if len(groups)
                else 0,
                "rows_in_near_duplicate_groups": len(groups),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / SUMMARY_OUTPUT_NAME, index=False)

    print(f"Input rows embedded: {len(frame)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Model: {MODEL_NAME}")
    print(f"Pairs >= {min(THRESHOLDS):.2f}: {len(pairs)}")
    print(summary.to_string(index=False))
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
