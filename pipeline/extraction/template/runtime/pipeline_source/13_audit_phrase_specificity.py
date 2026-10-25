#!/usr/bin/env python3
"""Reject underspecified final mechanism phrases and write a row-level audit."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


AUDIT_FIELDS = ["specificity_status", "specificity_rule", "specificity_normalized_phrase"]

GENERIC_ACTIONS = {
    "commercialize",
    "create",
    "develop",
    "distribute",
    "introduce",
    "launch",
    "manufacture",
    "market",
    "offer",
    "produce",
    "provide",
    "sell",
}
GENERIC_OBJECTS = {
    "business",
    "businesses",
    "goods",
    "operations",
    "offering",
    "offerings",
    "product",
    "products",
    "service",
    "services",
    "solution",
    "solutions",
}
DETERMINERS = {"a", "an", "the", "its", "our", "their"}
ASPECT_WORDS = {
    "begin",
    "began",
    "commence",
    "commenced",
    "continue",
    "continued",
    "start",
    "started",
}
VERB_SUFFIXES = ("ed", "es", "ing", "s")


def normalize_phrase(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"(?:\[\s*)?\bfocal[\s_-]+firm\b(?:\s*\])?", "[focal_firm]", text)
    text = re.sub(r"[^\w\[\]'-]+", " ", text)
    return " ".join(text.split())


def _base_verb(token: str) -> str:
    if token in GENERIC_ACTIONS:
        return token
    irregular = {"sold": "sell", "made": "manufacture", "provided": "provide"}
    if token in irregular:
        return irregular[token]
    for suffix in VERB_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            candidate = token[: -len(suffix)]
            candidates = {candidate}
            if suffix in {"ed", "es", "ing"}:
                candidates.add(candidate + "e")
            for value in candidates:
                if value in GENERIC_ACTIONS:
                    return value
    return token


def specificity_decision(phrase: str) -> tuple[bool, str, str]:
    """Return (keep, rule, normalized phrase) for a final mechanism phrase."""
    normalized = normalize_phrase(phrase)
    tokens = normalized.split()
    if not tokens:
        return False, "empty_phrase", normalized

    while tokens and tokens[0] in ASPECT_WORDS:
        tokens.pop(0)
    if tokens and tokens[0] in {"to", "will"}:
        tokens.pop(0)
    if not tokens:
        return False, "empty_after_normalization", normalized

    action = _base_verb(tokens[0])
    remainder = tokens[1:]
    while remainder and remainder[0] in DETERMINERS:
        remainder.pop(0)

    if action in GENERIC_ACTIONS and remainder == ["[focal_firm]"]:
        return False, "generic_action_on_focal_firm_placeholder", normalized
    if action in GENERIC_ACTIONS and len(remainder) == 1 and remainder[0] in GENERIC_OBJECTS:
        return False, "generic_action_on_generic_object", normalized
    return True, "specific_or_qualified", normalized


def audit_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    audited: list[dict[str, str]] = []
    for row in rows:
        phrase = row.get("final_mechanism_phrase", "") or row.get("atom_mechanism_phrase", "")
        keep, rule, normalized = specificity_decision(phrase)
        audited_row = dict(row)
        audited_row.update(
            {
                "specificity_status": "keep" if keep else "reject",
                "specificity_rule": rule,
                "specificity_normalized_phrase": normalized,
            }
        )
        audited.append(audited_row)
        if keep:
            kept.append(dict(row))
    return kept, audited


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kept-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Input has no header: {args.input}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    kept, audited = audit_rows(rows)
    write_csv(args.kept_output, kept, fields)
    write_csv(args.audit_output, audited, fields + AUDIT_FIELDS)

    counts: dict[str, int] = {}
    for row in audited:
        rule = row["specificity_rule"]
        counts[rule] = counts.get(rule, 0) + 1
    summary = {
        "status": "pass",
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rows) - len(kept),
        "rule_counts": counts,
        "kept_output": str(args.kept_output.resolve()),
        "audit_output": str(args.audit_output.resolve()),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary_output)
    print(
        f"Specificity audit kept {len(kept)}/{len(rows)} phrases; "
        f"rejected {len(rows) - len(kept)}."
    )


if __name__ == "__main__":
    main()
