#!/usr/bin/env python3
"""Export the five requested atom-phrase columns in a fixed order."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = [
        "ticker",
        "year",
        "atom_mechanism_phrase",
        "activity_phrase",
        "original_sentence",
    ]
    output = [
        {
            "ticker": row.get("ticker", ""),
            "year": row.get("year", ""),
            "atom_mechanism_phrase": (
                row.get("final_mechanism_phrase", "")
                or row.get("atom_mechanism_phrase", "")
            ),
            "activity_phrase": row.get("activity_phrase", ""),
            "original_sentence": row.get("source_sentence", ""),
        }
        for row in rows
    ]
    if any(not row["atom_mechanism_phrase"].strip() for row in output):
        raise SystemExit("Input contains an empty atom_mechanism_phrase")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)
    print(f"Exported {len(output)} atom phrases to {args.output}")


if __name__ == "__main__":
    main()
