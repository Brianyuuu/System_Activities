#!/usr/bin/env python3
"""Apply the frozen focal-business actor rule to any registrant."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


KEEP_CLASSES = {"focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate"}
KEEP_RELATIONS = {"focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate"}


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def is_self_alias(actor: str, ticker: str, company: str) -> bool:
    actor_norm = normalise(actor)
    if actor_norm in {"we", "our", "us", "the company", "the registrant", normalise(ticker)}:
        return True
    company_norm = normalise(company)
    stripped_company = re.sub(r"\b(?:incorporated|inc|corporation|corp|company|co|limited|ltd)\b", "", company_norm)
    stripped_actor = re.sub(r"\b(?:incorporated|inc|corporation|corp|company|co|limited|ltd)\b", "", actor_norm)
    return actor_norm == company_norm or bool(stripped_actor.strip() and stripped_actor.strip() == stripped_company.strip())


def classify(row: dict[str, str], ticker: str, company: str) -> tuple[str, str] | None:
    if row.get("transport_fallback", "").strip() == "source_clause":
        # The source text is preserved verbatim and must reach the strict
        # downstream audits; do not invent an actor classification here.
        return company, "stage01_transport_fallback"
    actor = row.get("resolved_actor_text", "").strip() or row.get("actor_text", "")
    actor_class = row.get("actor_class", "").strip().casefold()
    relation = row.get("actor_relation_to_focal", "").strip().casefold()
    if is_self_alias(actor, ticker, company):
        return company, "focal_firm"
    if actor_class in KEEP_CLASSES or relation in KEEP_RELATIONS:
        if "joint_venture_or_affiliate" in {actor_class, relation}:
            kind = "joint_venture_or_affiliate"
        elif "controlled_subsidiary" in {actor_class, relation}:
            kind = "controlled_subsidiary"
        else:
            kind = "focal_firm"
        return actor or company, kind
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company", required=True)
    args = parser.parse_args()
    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    additions = ("curated_actor", "curated_actor_relation", "curated_actor_filter", "curated_actor_resolution")
    fields += [field for field in additions if field not in fields]
    kept, rejected = [], []
    for row in rows:
        result = classify(row, args.ticker, args.company)
        if result:
            row["curated_actor"], row["curated_actor_relation"] = result
            row["curated_actor_filter"] = "keep"
            row["curated_actor_resolution"] = "focal_alias_or_extraction_relation"
            kept.append(row)
        else:
            row.update({"curated_actor": "", "curated_actor_relation": "", "curated_actor_filter": "reject", "curated_actor_resolution": ""})
            rejected.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("curated_candidates.csv", kept), ("rejected_candidates.csv", rejected), ("all_candidates.csv", rows)):
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(path)
    print(f"ticker={args.ticker} input={len(rows)} kept={len(kept)} rejected={len(rejected)}", flush=True)


if __name__ == "__main__":
    main()
