#!/usr/bin/env python3
"""Conservatively deduplicate same-meaning phrases from the same sentence."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PROTECTED = {
    "not", "no", "without", "only", "direct", "online", "in-house",
    "outsourced", "owned", "company-owned", "self-service", "exclusive",
    "domestic", "international", "regional", "global", "low-cost",
    "premium", "point-to-point", "hub-and-spoke", "free",
}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def protected_signature(text: str) -> tuple[str, ...]:
    phrase = str(text).lower()
    found = {term for term in PROTECTED if re.search(rf"\b{re.escape(term)}\b", phrase)}
    found.update(re.findall(r"\b\d[\d,.%/-]*\b", phrase))
    return tuple(sorted(found))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower()))


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def action_lemma(text: str, nlp) -> str:
    doc = nlp(str(text))
    for token in doc:
        if token.pos_ in {"VERB", "AUX"}:
            return token.lemma_.lower()
    return ""


def choose_representative(group: pd.DataFrame) -> int:
    scored = group.copy()
    scored["_both"] = scored["source_method"].str.contains("spacy") & scored["source_method"].str.contains("deepseek")
    scored["_llm"] = scored["source_method"].str.contains("deepseek")
    scored["_length"] = scored["activity_phrase_raw"].str.len()
    scored = scored.sort_values(
        ["_both", "_llm", "_length", "row_id"],
        ascending=[False, False, True, True],
    )
    return int(scored.index[0])


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project / "phrases/clean_normalize/no_bad_phrases/activities_medium_no_bad_phrases.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "phrases/clean_normalize/deduplication",
    )
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    frame["row_id"] = frame["row_id"].astype(int)
    frame = frame.reset_index(drop=True)
    nlp = spacy.load(args.spacy_model, disable=["ner"])
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(
        frame["activity_phrase_clean"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    actions = [action_lemma(value, nlp) for value in frame["activity_phrase_clean"]]
    signatures = [protected_signature(value) for value in frame["activity_phrase_clean"]]
    union_find = UnionFind(len(frame))
    pair_rows = []

    for _, subset in frame.groupby(["firm_id", "year", "source_sentence"], sort=False):
        indices = subset.index.tolist()
        for left, right in combinations(indices, 2):
            left_phrase = frame.at[left, "activity_phrase_clean"]
            right_phrase = frame.at[right, "activity_phrase_clean"]
            similarity = float(embeddings[left] @ embeddings[right])
            overlap = jaccard(left_phrase, right_phrase)
            exact = left_phrase == right_phrase
            same_action = bool(actions[left]) and actions[left] == actions[right]
            same_modifiers = signatures[left] == signatures[right]
            duplicate = exact or (
                same_modifiers
                and similarity >= 0.90
                and same_action
                and overlap >= 0.65
            )
            if duplicate:
                union_find.union(left, right)
            if similarity >= 0.80 or exact:
                pair_rows.append({
                    "left_row_id": frame.at[left, "row_id"],
                    "right_row_id": frame.at[right, "row_id"],
                    "firm_id": frame.at[left, "firm_id"],
                    "ticker": frame.at[left, "ticker"],
                    "source_sentence": frame.at[left, "source_sentence"],
                    "left_phrase": left_phrase,
                    "right_phrase": right_phrase,
                    "cosine_similarity": similarity,
                    "token_jaccard": overlap,
                    "left_action_lemma": actions[left],
                    "right_action_lemma": actions[right],
                    "protected_modifiers_match": same_modifiers,
                    "dedup_decision": "duplicate" if duplicate else "retain_both",
                })

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        components[union_find.find(index)].append(index)

    retained_rows = []
    dropped_rows = []
    group_number = 0
    for members in components.values():
        group = frame.loc[members]
        representative = choose_representative(group)
        if len(members) > 1:
            group_number += 1
            group_id = f"SSD{group_number:04d}"
        else:
            group_id = ""
        rep = frame.loc[representative].copy()
        rep["same_sentence_dedup_group_id"] = group_id
        rep["same_sentence_dedup_group_size"] = len(members)
        rep["dedup_member_row_ids"] = "|".join(str(frame.at[i, "row_id"]) for i in members)
        rep["dedup_member_phrases"] = " || ".join(frame.at[i, "activity_phrase_raw"] for i in members)
        rep["dedup_member_methods"] = "|".join(sorted(set(frame.loc[members, "source_method"])))
        rep["dedup_rule"] = "exact_or_conservative_embedding_same_sentence"
        retained_rows.append(rep)
        for index in members:
            if index == representative:
                continue
            dropped = frame.loc[index].copy()
            dropped["same_sentence_dedup_group_id"] = group_id
            dropped["retained_row_id"] = frame.at[representative, "row_id"]
            dropped["retained_activity_phrase_raw"] = frame.at[representative, "activity_phrase_raw"]
            dropped["retained_activity_phrase_clean"] = frame.at[representative, "activity_phrase_clean"]
            dropped["cosine_to_retained"] = float(embeddings[index] @ embeddings[representative])
            dropped["dedup_drop_reason"] = "same_sentence_same_meaning"
            dropped_rows.append(dropped)

    retained = pd.DataFrame(retained_rows).sort_values("row_id").reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows)
    pairs = pd.DataFrame(pair_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained.to_csv(args.output_dir / "activities_medium_deduplicated.csv", index=False)
    dropped.to_csv(args.output_dir / "same_sentence_duplicates_dropped.csv", index=False)
    pairs.to_csv(args.output_dir / "same_sentence_candidate_pairs.csv", index=False)
    np.save(args.output_dir / "deduplicated_activity_embeddings.npy", embeddings[[frame.index[frame.row_id.eq(rid)][0] for rid in retained.row_id]])
    summary = pd.DataFrame([
        {"metric": "input_rows", "value": len(frame)},
        {"metric": "retained_rows", "value": len(retained)},
        {"metric": "dropped_rows", "value": len(dropped)},
        {"metric": "duplicate_groups", "value": group_number},
        {"metric": "embedding_model", "value": args.embedding_model},
    ])
    summary.to_csv(args.output_dir / "deduplication_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
