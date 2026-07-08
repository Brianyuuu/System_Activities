#!/usr/bin/env python3
"""Create canonical sensitivity outputs and dependency-vs-LLM comparisons."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


FIRMS = ("delta", "ford", "southwest", "tesla")
SENSITIVITIES = ("strict", "medium", "loose")
TICKERS = {"delta": "DAL", "ford": "F", "southwest": "LUV", "tesla": "TSLA"}
SENSITIVITY_PATTERN = re.compile(r"(?:^|_)(strict|medium|loose)(?:_|\.)", re.I)

COMPARISON_FIELDS = [
    "firm_id",
    "ticker",
    "company",
    "year",
    "sensitivity",
    "method_match",
    "activity_phrase",
    "activity_phrase_clean",
    "source_sentence",
    "source_method",
    "chunk_id",
    "rule_score",
    "model_name",
    "prompt_version",
    "accession",
    "rules_triggered",
    "verb",
    "object",
    "subject",
    "subject_dependency",
    "sentence_index",
    "source_file",
    "dependency_row_count",
    "llm_row_count",
]

SUMMARY_FIELDS = [
    "firm_id",
    "year",
    "sensitivity",
    "n_dependency_only",
    "n_llm_only",
    "n_both",
    "n_total_unique",
    "n_dependency_rows",
    "n_llm_rows",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_sensitivity_outputs(source: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.iterdir()):
        if path.is_file() and SENSITIVITY_PATTERN.search(path.name):
            shutil.copy2(path, destination / path.name)
            copied += 1
    return copied


def distinct_join(values: list[str]) -> str:
    return "|".join(dict.fromkeys(value for value in values if value))


def group_rows(
    rows: list[dict[str, str]], key_columns: tuple[str, str]
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get(key_columns[0], ""), row.get(key_columns[1], ""))].append(row)
    return dict(grouped)


def values(rows: list[dict[str, str]], field: str) -> list[str]:
    return [row.get(field, "") for row in rows]


def comparison_row(
    firm: str,
    sensitivity: str,
    key: tuple[str, str],
    key_columns: tuple[str, str],
    dependency_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
) -> dict[str, Any]:
    all_rows = dependency_rows + llm_rows
    method_match = "both" if dependency_rows and llm_rows else (
        "dependency" if dependency_rows else "llm"
    )
    first = all_rows[0]
    activity_phrase = distinct_join(values(all_rows, "activity_phrase"))
    activity_phrase_clean = distinct_join(values(all_rows, "activity_phrase_clean"))
    if key_columns[0] == "activity_phrase_clean":
        activity_phrase_clean = key[0]
    else:
        activity_phrase = key[0]

    return {
        "firm_id": distinct_join(values(all_rows, "firm_id")),
        "ticker": TICKERS[firm],
        "company": distinct_join(values(all_rows, "company")),
        "year": distinct_join(values(all_rows, "year")),
        "sensitivity": sensitivity,
        "method_match": method_match,
        "activity_phrase": activity_phrase,
        "activity_phrase_clean": activity_phrase_clean,
        "source_sentence": key[1],
        "source_method": distinct_join(values(all_rows, "source_method")),
        "chunk_id": distinct_join(values(llm_rows, "chunk_id")),
        "rule_score": distinct_join(values(dependency_rows, "rule_score")),
        "model_name": distinct_join(values(llm_rows, "model") + values(llm_rows, "model_name")),
        "prompt_version": distinct_join(values(llm_rows, "prompt_version")),
        "accession": distinct_join(values(all_rows, "accession")),
        "rules_triggered": distinct_join(values(dependency_rows, "rules_triggered")),
        "verb": distinct_join(values(dependency_rows, "verb")),
        "object": distinct_join(values(dependency_rows, "object")),
        "subject": distinct_join(values(dependency_rows, "subject")),
        "subject_dependency": distinct_join(values(dependency_rows, "subject_dependency")),
        "sentence_index": distinct_join(values(dependency_rows, "sentence_index")),
        "source_file": distinct_join(values(all_rows, "source_file")),
        "dependency_row_count": len(dependency_rows),
        "llm_row_count": len(llm_rows),
    }


def build_comparison(
    firm: str,
    sensitivity: str,
    dependency_path: Path,
    llm_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    dependency, dependency_fields = read_csv(dependency_path)
    llm, llm_fields = read_csv(llm_path)
    phrase_key = (
        "activity_phrase_clean"
        if "activity_phrase_clean" in dependency_fields
        and "activity_phrase_clean" in llm_fields
        else "activity_phrase"
    )
    key_columns = (phrase_key, "source_sentence")
    dependency_grouped = group_rows(dependency, key_columns)
    llm_grouped = group_rows(llm, key_columns)
    keys = sorted(
        dependency_grouped.keys() | llm_grouped.keys(),
        key=lambda value: (value[1].casefold(), value[0].casefold()),
    )
    rows = [
        comparison_row(
            firm,
            sensitivity,
            key,
            key_columns,
            dependency_grouped.get(key, []),
            llm_grouped.get(key, []),
        )
        for key in keys
    ]
    write_csv(output_path, rows, COMPARISON_FIELDS)

    counts = {label: sum(row["method_match"] == label for row in rows) for label in ("dependency", "llm", "both")}
    return {
        "firm_id": rows[0]["firm_id"] if rows else "",
        "year": rows[0]["year"] if rows else "2015",
        "sensitivity": sensitivity,
        "n_dependency_only": counts["dependency"],
        "n_llm_only": counts["llm"],
        "n_both": counts["both"],
        "n_total_unique": len(rows),
        "n_dependency_rows": len(dependency),
        "n_llm_rows": len(llm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_dir
    phrases = project / "phrases"
    dependency_destination = phrases / "dependency_phrases"
    llm_destination = phrases / "llm_phrases"
    comparison_destination = phrases / "comparison"

    dependency_count = copy_sensitivity_outputs(
        project / "dependency_phrases", dependency_destination
    )
    llm_count = copy_sensitivity_outputs(project / "llm_phrases", llm_destination)

    summaries: list[dict[str, Any]] = []
    for firm in FIRMS:
        for sensitivity in SENSITIVITIES:
            summaries.append(
                build_comparison(
                    firm,
                    sensitivity,
                    dependency_destination / f"{firm}_{sensitivity}_2015.csv",
                    llm_destination / f"{firm}_deepseek_{sensitivity}_2015.csv",
                    comparison_destination
                    / f"{firm}_{sensitivity}_2015_comparison.csv",
                )
            )

    write_csv(
        comparison_destination / "comparison_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    print(
        f"Copied {dependency_count} dependency and {llm_count} LLM sensitivity files; "
        f"created {len(summaries)} comparisons."
    )


if __name__ == "__main__":
    main()
