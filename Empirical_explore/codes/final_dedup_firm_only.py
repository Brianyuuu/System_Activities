#!/usr/bin/env python3
"""Create a final firm-only activity file with exact and semantic duplicates removed.

Input:
  phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only.csv
  phrases/clean_normalize/deduplication/deepseek_semantic_duplicate_pairs.csv

Output:
  phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv
  phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_dedup_dropped.csv
  phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_dedup_summary.csv
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path("/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore")
ACTOR_DIR = PROJECT_DIR / "phrases/clean_normalize/actor_identification"
DEDUP_DIR = PROJECT_DIR / "phrases/clean_normalize/deduplication"

INPUT_FIRM_ONLY = ACTOR_DIR / "activities_medium_actor_firm_only.csv"
INPUT_DEEPSEEK_PAIRS = DEDUP_DIR / "deepseek_semantic_duplicate_pairs.csv"

OUTPUT_RETAINED = ACTOR_DIR / "activities_medium_actor_firm_only_deduplicated.csv"
OUTPUT_DROPPED = ACTOR_DIR / "activities_medium_actor_firm_only_dedup_dropped.csv"
OUTPUT_SUMMARY = ACTOR_DIR / "activities_medium_actor_firm_only_dedup_summary.csv"


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: object) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def jaccard(a: object, b: object) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {v: v for v in values}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def source_method_score(value: object) -> int:
    s = str(value).lower()
    if "both" in s:
        return 4
    if "deepseek" in s or "llm" in s:
        return 3
    if "spacy" in s or "dependency" in s:
        return 2
    return 1


def phrase_len_score(value: object) -> int:
    """Prefer informative, moderately detailed phrases.

    Activity-system coding benefits from retaining object/mode/scope details, so
    longer phrases usually beat very short variants. Very long phrases get only
    a small penalty through the final row_id tie-breaker rather than hard removal.
    """

    return len(TOKEN_RE.findall(str(value).lower()))


def main() -> None:
    df = pd.read_csv(INPUT_FIRM_ONLY)
    pairs = pd.read_csv(INPUT_DEEPSEEK_PAIRS)

    df["row_id_str"] = df["row_id"].astype(str)
    row_ids = df["row_id_str"].tolist()
    id_set = set(row_ids)

    uf = UnionFind(row_ids)
    edge_rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    canonical_by_row: dict[str, list[str]] = defaultdict(list)

    def add_edge(left: object, right: object, rule: str) -> None:
        a, b = str(left), str(right)
        if a == b or a not in id_set or b not in id_set:
            return
        uf.union(a, b)
        edge_rules[tuple(sorted((a, b)))].add(rule)

    # 1) Exact duplicates after final firm-only filtering.
    exact_keys = ["ticker", "year", "activity_phrase_clean"]
    for _, group in df.groupby(exact_keys, dropna=False):
        ids = group["row_id_str"].tolist()
        if len(ids) <= 1:
            continue
        first = ids[0]
        for other in ids[1:]:
            add_edge(first, other, "exact_activity_phrase_clean")

    # 2) Residual semantic duplicates labeled by DeepSeek.
    duplicate_pairs = pairs[pairs["deepseek_duplicate_label"].astype(str).str.lower().eq("duplicate")]
    for _, row in duplicate_pairs.iterrows():
        left = str(row["left_row_id"])
        right = str(row["right_row_id"])
        if left in id_set and right in id_set:
            add_edge(left, right, "deepseek_semantic_duplicate")
            canonical = str(row.get("deepseek_canonical_phrase", "")).strip()
            if canonical:
                canonical_by_row[left].append(canonical)
                canonical_by_row[right].append(canonical)

    components: dict[str, list[str]] = defaultdict(list)
    for rid in row_ids:
        components[uf.find(rid)].append(rid)

    component_rows = []
    retained_ids: set[str] = set()

    df_by_id = df.set_index("row_id_str", drop=False)

    for comp_index, ids in enumerate(
        sorted(components.values(), key=lambda x: min(int(i) for i in x)),
        start=1,
    ):
        group_id = f"FFD{comp_index:04d}"
        member_phrases = [str(df_by_id.loc[rid, "activity_phrase_clean"]) for rid in ids]
        member_methods = [str(df_by_id.loc[rid, "source_method"]) for rid in ids]

        group_rules: set[str] = set()
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                group_rules.update(edge_rules.get(tuple(sorted((left, right))), set()))
        if not group_rules and len(ids) == 1:
            group_rules.add("unique")

        def representative_score(rid: str) -> tuple[float, int, int, int]:
            row = df_by_id.loc[rid]
            canonicals = canonical_by_row.get(rid, [])
            canonical_score = max([jaccard(row["activity_phrase_clean"], c) for c in canonicals] or [0.0])
            return (
                canonical_score,
                source_method_score(row["source_method"]),
                phrase_len_score(row["activity_phrase_clean"]),
                -int(row["row_id"]),
            )

        kept_id = max(ids, key=representative_score)
        retained_ids.add(kept_id)

        kept_reason = (
            "unique row"
            if len(ids) == 1
            else "highest score on DeepSeek canonical overlap, source-method priority, and phrase detail"
        )

        for rid in ids:
            component_rows.append(
                {
                    "row_id_str": rid,
                    "final_firm_dedup_group_id": group_id,
                    "final_firm_dedup_group_size": len(ids),
                    "final_firm_dedup_member_row_ids": "|".join(ids),
                    "final_firm_dedup_member_phrases": " || ".join(member_phrases),
                    "final_firm_dedup_member_methods": "|".join(member_methods),
                    "final_firm_dedup_rules": "|".join(sorted(group_rules)),
                    "final_firm_dedup_representative_row_id": kept_id,
                    "final_firm_dedup_kept": rid == kept_id,
                    "final_firm_dedup_kept_reason": kept_reason,
                }
            )

    meta = pd.DataFrame(component_rows)
    out = df.merge(meta, on="row_id_str", how="left").drop(columns=["row_id_str"])

    retained = out[out["final_firm_dedup_kept"]].copy()
    dropped = out[~out["final_firm_dedup_kept"]].copy()

    retained.to_csv(OUTPUT_RETAINED, index=False)
    dropped.to_csv(OUTPUT_DROPPED, index=False)

    exact_groups_before = (
        df.groupby(["ticker", "year", "activity_phrase_clean"], dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    exact_groups_after = (
        retained.groupby(["ticker", "year", "activity_phrase_clean"], dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )

    retained_id_set = set(retained["row_id"].astype(str))
    semantic_pairs_before = duplicate_pairs[
        duplicate_pairs["left_row_id"].astype(str).isin(id_set)
        & duplicate_pairs["right_row_id"].astype(str).isin(id_set)
    ]
    semantic_pairs_after = duplicate_pairs[
        duplicate_pairs["left_row_id"].astype(str).isin(retained_id_set)
        & duplicate_pairs["right_row_id"].astype(str).isin(retained_id_set)
    ]

    summary = pd.DataFrame(
        [
            {"metric": "input_firm_only_rows", "value": len(df)},
            {"metric": "output_retained_rows", "value": len(retained)},
            {"metric": "dropped_duplicate_rows", "value": len(dropped)},
            {"metric": "final_dedup_groups_total", "value": len(components)},
            {"metric": "final_dedup_groups_size_gt_1", "value": sum(len(v) > 1 for v in components.values())},
            {"metric": "exact_duplicate_groups_before", "value": len(exact_groups_before)},
            {"metric": "exact_duplicate_groups_after", "value": len(exact_groups_after)},
            {"metric": "deepseek_duplicate_pairs_before", "value": len(semantic_pairs_before)},
            {"metric": "deepseek_duplicate_pairs_after", "value": len(semantic_pairs_after)},
        ]
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"Input firm-only rows: {len(df)}")
    print(f"Retained rows: {len(retained)}")
    print(f"Dropped duplicate rows: {len(dropped)}")
    print(f"Duplicate groups size > 1: {sum(len(v) > 1 for v in components.values())}")
    print(f"Exact duplicate groups before/after: {len(exact_groups_before)} / {len(exact_groups_after)}")
    print(f"DeepSeek duplicate pairs before/after: {len(semantic_pairs_before)} / {len(semantic_pairs_after)}")
    print(f"Saved: {OUTPUT_RETAINED}")
    print(f"Saved: {OUTPUT_DROPPED}")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
