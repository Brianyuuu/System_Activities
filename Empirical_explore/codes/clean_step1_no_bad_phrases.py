#!/usr/bin/env python3
"""Step 1 cleaning: combine medium candidates, normalize, remove obvious fragments.

This stage intentionally does not judge actor validity, activity-vs-resource
status, repeatability, granularity, or semantic duplication. Those decisions
belong to later cleaning stages.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy


PRESERVE_STOPWORDS = {
    # These words carry activity-system mode, scope, channel, or negation.
    "against", "by", "direct", "during", "for", "from", "in", "into",
    "not", "no", "only", "over", "through", "to", "under", "using",
    "via", "with", "without",
}

CANONICAL_COMPOUNDS = {
    r"\bpoint\s*-?\s*to\s*-?\s*point\b": "point-to-point",
    r"\bhub\s*-?\s*and\s*-?\s*spoke\b": "hub-and-spoke",
    r"\blow\s*-?\s*cost\b": "low-cost",
    r"\bsame\s*-?\s*day\b": "same-day",
    r"\bstore\s*-?\s*based\b": "store-based",
    r"\bin\s*-?\s*house\b": "in-house",
    r"\bself\s*-?\s*service\b": "self-service",
    r"\bcompany\s*-?\s*owned\b": "company-owned",
}

FRAGMENT_PATTERNS = (
    re.compile(
        r"^(?:operate|serve|continue|use|provide|be|sell|offer|incur) "
        r"(?:that|which|it|there)\b",
        re.I,
    ),
    re.compile(r"\b(?:that|which) to$", re.I),
    re.compile(r"^(?:receive for(?: which)?(?: .*)?|operate in future|select for|serve for .*)$", re.I),
)


def canonicalize(text: str) -> str:
    text = str(text).replace("–", "-").replace("—", "-").replace("‑", "-")
    for pattern, replacement in CANONICAL_COMPOUNDS.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def normalize_phrase(text: str, nlp) -> str:
    doc = nlp(canonicalize(text).lower())
    output: list[str] = []
    for token in doc:
        raw = token.text.lower().strip()
        if not raw or token.is_punct or raw in {"'s", "’s"}:
            continue
        if token.is_stop and raw not in PRESERVE_STOPWORDS:
            continue
        value = raw if raw in PRESERVE_STOPWORDS or "-" in raw else token.lemma_.lower().strip()
        if value:
            output.append(value)
    return canonicalize(" ".join(output)).strip(" ,;:-")


def obvious_bad_reason(cleaned: str, nlp) -> str:
    if not cleaned:
        return "empty_after_normalization"
    tokens = cleaned.split()
    if len(tokens) == 1:
        return "single_token_phrase"
    doc = nlp(cleaned)
    content_count = sum(
        token.pos_ in {"VERB", "NOUN", "PROPN", "ADJ", "ADV"}
        for token in doc
        if not token.is_punct
    )
    if content_count < 2:
        return "insufficient_content"
    if any(pattern.search(cleaned) for pattern in FRAGMENT_PATTERNS):
        return "syntactic_fragment"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=project / "phrases/comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "phrases/clean_normalize/no_bad_phrases",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.comparison_dir.glob("*_medium_2015_comparison.csv"))
    if len(paths) != 4:
        raise SystemExit(f"Expected four medium comparison files; found {len(paths)}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.insert(0, "input_file", path.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "row_id", np.arange(len(combined), dtype=int))
    combined["activity_phrase_raw"] = combined["activity_phrase"]

    nlp = spacy.load(args.spacy_model, disable=["ner"])
    combined["activity_phrase_clean"] = [
        normalize_phrase(value, nlp) for value in combined["activity_phrase_raw"]
    ]
    combined["bad_phrase_reason"] = [
        obvious_bad_reason(value, nlp) for value in combined["activity_phrase_clean"]
    ]
    combined["step1_status"] = np.where(
        combined["bad_phrase_reason"].eq(""), "retained", "excluded"
    )

    retained = combined[combined["step1_status"].eq("retained")].copy()
    excluded = combined[combined["step1_status"].eq("excluded")].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "activities_medium_step1_all.csv", index=False)
    retained.to_csv(
        args.output_dir / "activities_medium_no_bad_phrases.csv", index=False
    )
    excluded.to_csv(
        args.output_dir / "excluded_obvious_bad_phrases.csv", index=False
    )

    summary_rows = [
        {"section": "overall", "category": "input_rows", "count": len(combined)},
        {"section": "overall", "category": "retained_rows", "count": len(retained)},
        {"section": "overall", "category": "excluded_rows", "count": len(excluded)},
    ]
    for reason, count in excluded["bad_phrase_reason"].value_counts().items():
        summary_rows.append(
            {"section": "excluded_reason", "category": reason, "count": int(count)}
        )
    for ticker, count in retained["ticker"].value_counts().items():
        summary_rows.append(
            {"section": "retained_by_firm", "category": ticker, "count": int(count)}
        )
    for method, count in retained["source_method"].value_counts().items():
        summary_rows.append(
            {"section": "retained_by_method", "category": method, "count": int(count)}
        )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "step1_cleaning_summary.csv", index=False
    )

    print(f"Input rows: {len(combined):,}")
    print(f"Retained rows: {len(retained):,}")
    print(f"Excluded rows: {len(excluded):,}")
    print(excluded["bad_phrase_reason"].value_counts().to_string())
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
