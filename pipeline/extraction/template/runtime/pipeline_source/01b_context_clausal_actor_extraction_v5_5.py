#!/usr/bin/env python3
"""Resumable all-airline v5.4 DeepSeek extraction with audited Item 1 boundaries.

The v5.4 primary and repair prompts, clause construction, and response
validation are intentionally unchanged.  v5.5 replaces the fragile
filing-name/boundary layer and adds request-level checkpointing.  Optional
transport recovery bisects an invalid repair response while retaining the
original prompt and source IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import spacy
from openai import OpenAI

from item1_boundary_v5_5 import extract_item1, html_to_text


# Kept local so the frozen benchmark workflow does not depend on the historical
# D:\codex\dal_actor_global_v3 checkout.  These values and helpers are byte-for-
# byte equivalent in behavior to the pieces formerly imported from
# actor_first_pipeline_extraction.py and extract_replaced_phrases.py.
CORE_FIELDS = [
    "decision", "activity_phrase", "source_sentence", "actor_text", "actor_class",
    "actor_relation_to_focal", "relationship_evidence", "is_focal_firm_activity", "mechanism_text",
]
METADATA_FIELDS = [
    "firm_id", "ticker", "company", "year", "accession", "model", "source_file", "chunk_id",
]
ACTOR_CLASSES = {
    "focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate", "contracted_provider",
    "partner", "customer", "supplier", "regulator", "product_or_system", "external_actor", "unclear",
}
FOCAL_VALUES = {"yes", "no", "unclear"}
ACTOR_RELATIONS = {
    "focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate", "contracted_provider",
    "partner", "customer", "supplier", "regulator", "unrelated_external", "unclear",
}


def split_prompt(path: Path) -> tuple[str | None, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    if "SYSTEM PROMPT" in text and "USER PROMPT" in text:
        system, user = text.split("USER PROMPT", 1)
        return system.replace("SYSTEM PROMPT", "", 1).strip(), user.strip()
    return None, text


def make_chunks(sentences: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_chars + added > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += added
    if current:
        chunks.append(current)
    return chunks


def build_request_messages(
    template: str,
    focal_firm: str,
    records: list[dict[str, str]],
    cache_optimized: bool,
) -> list[dict[str, str]]:
    """Keep a stable DeepSeek prefix without changing extraction rules.

    The legacy format substitutes the firm name near the first line, which
    prevents automatic prompt-cache reuse across firms.  The opt-in media
    format preserves the complete instruction template as a stable system
    prefix and supplies only the two placeholder values in the user payload.
    """
    records_json = json.dumps(records, ensure_ascii=False)
    if not cache_optimized:
        prompt = template.replace("{clause_records_json}", records_json).replace(
            "{focal_firm}", focal_firm
        )
        return [{"role": "user", "content": prompt}]
    stable_template = template.replace(
        "{focal_firm}", "the focal firm named in FOCAL_FIRM in the request payload"
    ).replace(
        "{clause_records_json}", "the JSON array named CLAUSE_RECORDS in the request payload"
    )
    dynamic_payload = json.dumps(
        {"FOCAL_FIRM": focal_firm, "CLAUSE_RECORDS": records},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": stable_template},
        {"role": "user", "content": dynamic_payload},
    ]


RJET_COMPANY_BY_YEAR = {
    "mesa": "Mesa Air Group, Inc.",
    "republic": "Republic Airways Holdings Inc.",
}

FOCAL_BUSINESS_CLASSES = {"focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate"}
FOCAL_BUSINESS_RELATIONS = {"focal_firm", "controlled_subsidiary", "joint_venture_or_affiliate"}


def canonical_company(row: dict[str, str]) -> tuple[str, str]:
    """Correct the known CIK/ticker alias error without changing source IDs."""
    ticker, year = row["ticker"].upper(), int(row["report_year"])
    if ticker == "RJET":
        if year < 2025:
            return RJET_COMPANY_BY_YEAR["mesa"], "manifest_RJET_label_corrected_to_Mesa"
        return RJET_COMPANY_BY_YEAR["republic"], ""
    return row["company"].strip(), ""


def split_clauses(sentence: str, threshold: int) -> list[str]:
    if len(sentence) <= threshold:
        return [sentence]
    parts = re.split(
        r"(?<=[;:])\s+|(?<=[.!?])\s+(?=[A-Z])|,\s+(?=(?:and|but|while|which|that|as|including)\b)",
        sentence,
    )
    return [part.strip() for part in parts if part.strip()]


def _load_canonical_item1(path: Path) -> str:
    """Read the single, upstream-audited Item 1 used by every later stage."""

    text = path.read_text(encoding="utf-8-sig")
    marker = "\n# Item 1. Business\n\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


def prepare(
    path_text: str,
    metadata: dict[str, str],
    chunk_chars: int,
    item1_min_words: int = 2_000,
    item1_warning_words: int = 3_000,
    clean_item1_path_text: str = "",
) -> dict[str, Any]:
    path = Path(path_text)
    try:
        clean_item1_path = Path(clean_item1_path_text) if clean_item1_path_text else None
        if clean_item1_path is not None and clean_item1_path.is_file():
            # The cleaning stage has already isolated the primary SEC document,
            # preserved layout, rejected TOC/cross-reference boundaries, and
            # written the canonical Item 1.  Re-extracting from the complete
            # submission here created two competing answers for the same filing.
            section = _load_canonical_item1(clean_item1_path)
            section_words = len(section.split())
            if section_words < item1_min_words:
                raise ValueError(
                    f"Canonical Item 1 is too short for {path.name}: "
                    f"words={section_words} minimum={item1_min_words}"
                )
            audit_data = {
                "source_file": path.name,
                "visible_words": section_words,
                "section_words": section_words,
                "start_char": -1,
                "end_char": -1,
                "start_text": "",
                "end_text": "",
                "start_candidates": 1,
                "end_candidates": 1,
                "rejected_toc_starts": 0,
                "rejected_cross_reference_starts": 0,
                "rejected_cross_reference_ends": 0,
                "warning": "canonical_clean_item1",
            }
        else:
            visible = html_to_text(path)
            section, audit = extract_item1(
                visible,
                source_file=path.name,
                min_words=item1_min_words,
                warning_words=item1_warning_words,
            )
            audit_data = audit.to_dict()
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 5_000_000
        # Full-category input guard. Prompt, clause splitting and downstream logic unchanged.
        segments = [section]
        if clean_item1_path is not None and clean_item1_path.is_file():
            canonical_text = clean_item1_path.read_text(encoding="utf-8-sig")
            metadata_lines = [line for line in canonical_text.splitlines()[:4] if line.startswith("metadata: ")]
            source_meta = json.loads(metadata_lines[0][10:]) if metadata_lines else {}
            spans = source_meta.get("source_part_spans", [])
            if spans:
                segments = [section[int(a):int(b)] for a, b in spans]
                if "\n\n".join(segments) != section:
                    raise ValueError("Source-part spans do not reconstruct canonical Item 1")
        chunks = []
        for segment in segments:
            sentences = [sentence.text.strip() for sentence in nlp(segment).sents if sentence.text.strip()]
            chunks.extend(make_chunks(sentences, chunk_chars))
        audit_data["source_part_count"] = len(segments)
        return {
            "path": path_text,
            "metadata": metadata,
            "chunks": chunks,
            "boundary": {**audit_data, "status": "ok", "error": "", **metadata},
        }
    except Exception as error:  # returned for an auditable fail-fast manifest
        return {
            "path": path_text,
            "metadata": metadata,
            "chunks": [],
            "boundary": {
                "source_file": path.name,
                "status": "error",
                "error": repr(error),
                "warning": "",
                **metadata,
            },
        }


def records_for_chunk(sentences: list[str], threshold: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    number = 0
    for index, sentence in enumerate(sentences):
        preceding = sentences[index - 1] if index else ""
        prior_clauses: list[str] = []
        for clause in split_clauses(sentence, threshold):
            number += 1
            context = " ".join(part for part in [preceding, *prior_clauses] if part)
            records.append({"source_id": f"C{number:04d}", "context": context, "target_clause": clause})
            prior_clauses.append(clause)
    return records


def validate(
    payload: dict[str, Any], source_by_id: dict[str, str], context_by_id: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    values = payload.get("activities") if isinstance(payload, dict) else None
    stats = {
        "raw": len(values) if isinstance(values, list) else 0,
        "not_list": 0,
        "bad_source_id": 0,
        "bad_decision": 0,
        "missing_phrase": 0,
        "valid": 0,
    }
    output: list[dict[str, str]] = []
    if not isinstance(values, list):
        stats["not_list"] = 1
        return output, stats
    for item in values:
        if not isinstance(item, dict):
            stats["not_list"] += 1
            continue
        source_id = str(item.get("source_id", "")).strip()
        decision = str(item.get("decision", "")).strip().lower()
        phrase = str(item.get("activity_phrase", "")).strip()
        if source_id not in source_by_id:
            stats["bad_source_id"] += 1
            continue
        if decision not in {"keep", "reject"}:
            stats["bad_decision"] += 1
            continue
        if not phrase:
            stats["missing_phrase"] += 1
            continue
        actor_class = str(item.get("actor_class", "unclear")).strip()
        relation = str(item.get("actor_relation_to_focal", "unclear")).strip()
        focal = str(item.get("is_focal_firm_activity", "unclear")).strip().lower()
        if actor_class in FOCAL_BUSINESS_CLASSES or relation in FOCAL_BUSINESS_RELATIONS:
            focal = "yes"
        output.append(
            {
                "decision": decision,
                "activity_phrase": phrase,
                "source_sentence": source_by_id[source_id],
                "prior_context": context_by_id[source_id],
                "actor_text": str(item.get("actor_surface_text", item.get("actor_text", ""))).strip(),
                "resolved_actor_text": str(item.get("resolved_actor_text", "")).strip(),
                "actor_resolution_status": str(item.get("actor_resolution_status", "not_needed")).strip().lower(),
                "actor_class": actor_class if actor_class in ACTOR_CLASSES else "unclear",
                "actor_relation_to_focal": relation if relation in ACTOR_RELATIONS else "unclear",
                "relationship_evidence": str(item.get("relationship_evidence", "")).strip(),
                "is_focal_firm_activity": focal if focal in FOCAL_VALUES else "unclear",
                "mechanism_text": str(item.get("mechanism_text", "")).strip(),
            }
        )
        stats["valid"] += 1
    return output, stats


def response_is_structurally_valid(diagnostic: dict[str, Any]) -> bool:
    """Return true for a schema-valid response, including activities=[]."""
    stats = diagnostic.get("validation", {})
    invalid = sum(
        int(stats.get(field, 0) or 0)
        for field in ("not_list", "bad_source_id", "bad_decision", "missing_phrase")
    )
    return not diagnostic.get("error") and invalid == 0


def transport_fallback_activity(record: dict[str, str]) -> dict[str, str]:
    """Preserve an unresolved source clause for the downstream strict audits."""
    return {
        "decision": "keep",
        "activity_phrase": record["target_clause"],
        "source_sentence": record["target_clause"],
        "prior_context": record.get("context", ""),
        "actor_text": "",
        "resolved_actor_text": "",
        "actor_resolution_status": "unresolved",
        "actor_class": "unclear",
        "actor_relation_to_focal": "unclear",
        "relationship_evidence": record["target_clause"],
        "is_focal_firm_activity": "unclear",
        "mechanism_text": "",
        "transport_fallback": "source_clause",
        "transport_fallback_reason": "stage01_api_or_schema_failure_after_bisection",
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--repair-prompt-file", type=Path, required=True)
    parser.add_argument("--firms", default="ALL")
    parser.add_argument("--min-year", type=int, default=2016)
    parser.add_argument("--max-year", type=int, default=9999)
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--clause-threshold", type=int, default=350)
    parser.add_argument("--item1-min-words", type=int, default=2000)
    parser.add_argument("--item1-warning-words", type=int, default=3000)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-workers", type=int, default=256)
    parser.add_argument("--submission-batch-size", type=int, default=256)
    parser.add_argument("--cpu-workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument(
        "--cache-optimized-prompt-format",
        action="store_true",
        help=(
            "Place invariant rules in a stable system prefix and dynamic focal-firm/records "
            "in the user payload so DeepSeek automatic prompt caching can be reused."
        ),
    )
    parser.add_argument(
        "--allow-boundary-warnings",
        action="store_true",
        help="Continue after audited boundary warnings; boundary errors still fail.",
    )
    parser.add_argument(
        "--bisect-on-failure",
        action="store_true",
        help="Bisect unresolved repair requests while preserving source IDs and prompts.",
    )
    parser.add_argument(
        "--fallback-to-source-on-failure",
        action="store_true",
        help="Keep an unresolved singleton source clause for downstream strict audits.",
    )
    args = parser.parse_args()

    _, primary_template = split_prompt(args.prompt_file)
    _, repair_template = split_prompt(args.repair_prompt_file)
    selected_firms = {item.strip().upper() for item in args.firms.split(",") if item.strip()}
    all_firms = selected_firms == {"ALL"}

    with args.manifest_csv.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    inputs: list[tuple[Path, dict[str, str], Path | None]] = []
    skipped: list[dict[str, Any]] = []
    for source in manifest_rows:
        year = int(source["report_year"])
        ticker = source["ticker"].upper()
        form = source["form"].upper()
        if not (args.min_year <= year <= args.max_year) or (not all_firms and ticker not in selected_firms):
            continue
        if (form.endswith("/A") or form.endswith("-A")) and source["accession_number"] not in {'0001045969-98-000859', '0000950136-98-000983', '0000950136-02-001827', '0001193125-14-368900', '0001047469-04-022965'}:
            skipped.append({**source, "status": "skipped_amendment_without_full_item1"})
            continue
        # Historical EDGAR filings use 10-K405, transition-form 10KT405,
        # and small-business variants. The upstream canonical Item 1
        # extractor has already enforced the allowed-form and boundary rules.
        if (form[:-2] if form.endswith("/A") else form) not in {"10-K", "10-KT", "10-K405", "10KT405", "10KSB", "10-KSB", "10-KSB40", "10KSB40"}:
            skipped.append({**source, "status": "skipped_non_10k"})
            continue
        path = args.input_dir / source["input_file"]
        company, metadata_warning = canonical_company(source)
        metadata = {
            "firm_id": ticker,
            "ticker": ticker,
            "company": company,
            "year": str(year),
            "accession": source["accession_number"],
            "form": form,
            "filing_date": source["filing_date"],
            "metadata_warning": metadata_warning,
        }
        clean_item1_name = source.get("clean_item1_file", "").strip()
        clean_item1_path = args.input_dir / clean_item1_name if clean_item1_name else None
        inputs.append((path, metadata, clean_item1_path))
    if not inputs:
        raise SystemExit("No filings matched the requested firms/years/forms")

    prepared: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.cpu_workers)) as pool:
        futures = [
            pool.submit(
                prepare,
                str(path),
                metadata,
                args.chunk_chars,
                args.item1_min_words,
                args.item1_warning_words,
                str(clean_item1_path) if clean_item1_path is not None else "",
            )
            for path, metadata, clean_item1_path in inputs
        ]
        for number, future in enumerate(as_completed(futures), 1):
            prepared.append(future.result())
            if number % 10 == 0 or number == len(futures):
                print(f"Prepared {number}/{len(futures)} filings", flush=True)

    prepared.sort(key=lambda item: item["path"])
    boundary_rows = [item["boundary"] for item in prepared]
    boundary_fields = sorted({key for row in boundary_rows for key in row})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "item1_boundary_audit.csv", boundary_rows, boundary_fields)
    atomic_json(args.output_dir / "skipped_manifest_rows.json", skipped)
    boundary_errors = [row for row in boundary_rows if row["status"] != "ok"]
    boundary_warnings = [row for row in boundary_rows if row.get("warning")]
    if boundary_errors or (boundary_warnings and not args.allow_boundary_warnings):
        raise SystemExit(
            f"Boundary preflight failed: errors={len(boundary_errors)} warnings={len(boundary_warnings)}; "
            f"see {args.output_dir / 'item1_boundary_audit.csv'}"
        )

    jobs: list[dict[str, Any]] = []
    for item in prepared:
        for chunk_id, chunk in enumerate(item["chunks"], 1):
            jobs.append(
                {
                    "path": Path(item["path"]),
                    "metadata": item["metadata"],
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                }
            )
    print(
        json.dumps(
            {
                "filings": len(prepared),
                "requests": len(jobs),
                "skipped_amendments": sum(row["status"].startswith("skipped_amendment") for row in skipped),
                "model": args.model,
                "boundary_only": args.boundary_only,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.boundary_only:
        return

    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    client = OpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=300.0,
    )
    cache_dir = args.output_dir / "response_cache"

    def run(
        job: dict[str, Any], repair: bool = False,
        records_override: list[dict[str, str]] | None = None,
        cache_tag: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
        records = records_override if records_override is not None else records_for_chunk(
            job["chunk"], args.clause_threshold
        )
        source_by_id = {record["source_id"]: record["target_clause"] for record in records}
        context_by_id = {record["source_id"]: record["context"] for record in records}
        template = repair_template if repair else primary_template
        messages = build_request_messages(
            template,
            job["metadata"]["company"],
            records,
            args.cache_optimized_prompt_format,
        )
        pass_name = "repair" if repair else "primary"
        request_hash = hashlib.sha256(
            json.dumps(
                {"model": args.model, "messages": messages,
                 "response_format": "glm_json_schema_v1" if args.model == "parcc/GLM" else "json_object"},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = cache_dir / (
            f"{job['path'].stem}__chunk{job['chunk_id']:04d}__{pass_name}{cache_tag}.json"
        )
        raw, error, attempts, cached = "", "", 0, False
        if cache_path.exists():
            try:
                cached_value = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached_value.get("request_hash") == request_hash and cached_value.get("raw_response"):
                    raw = cached_value["raw_response"]
                    cached = True
            except (OSError, ValueError, TypeError):
                pass
        if not raw:
            last_error: Exception | None = None
            for attempt in range(args.max_retries + 1):
                attempts = attempt + 1
                try:
                    activity_properties = {
                        "source_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["keep", "reject"]},
                        "activity_phrase": {"type": "string"},
                        "actor_surface_text": {"type": "string"},
                        "resolved_actor_text": {"type": "string"},
                        "actor_resolution_status": {"type": "string", "enum": ["not_needed", "resolved_from_context", "unresolved"]},
                        "actor_class": {"type": "string", "enum": sorted(ACTOR_CLASSES)},
                        "actor_relation_to_focal": {"type": "string", "enum": sorted(ACTOR_RELATIONS)},
                        "relationship_evidence": {"type": "string"},
                        "is_focal_firm_activity": {"type": "string", "enum": sorted(FOCAL_VALUES)},
                        "mechanism_text": {"type": "string"},
                    }
                    response_format = (
                        {"type": "json_schema", "json_schema": {
                            "name": "actor_activities", "strict": True,
                            "schema": {"type": "object", "additionalProperties": False,
                                       "properties": {"activities": {"type": "array", "items": {
                                           "type": "object", "additionalProperties": False,
                                           "properties": activity_properties,
                                           "required": list(activity_properties),
                                       }}}, "required": ["activities"]},
                        }} if args.model == "parcc/GLM" else {"type": "json_object"}
                    )
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        response_format=response_format,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        extra_body={
                            "thinking": {"type": "disabled"},
                            **({"chat_template_kwargs": {"enable_thinking": False}} if args.model == "parcc/GLM" else {}),
                        },
                    )
                    message = response.choices[0].message
                    raw = message.content or (message.model_extra or {}).get("reasoning_content") or "{}"
                    json.loads(raw)
                    atomic_json(
                        cache_path,
                        {
                            "request_hash": request_hash,
                            "model": args.model,
                            "source_file": job["path"].name,
                            "chunk_id": job["chunk_id"],
                            "pass": pass_name,
                            "cache_optimized_prompt_format": args.cache_optimized_prompt_format,
                            "raw_response": raw,
                        },
                    )
                    break
                except Exception as exc:  # API, timeout, or malformed JSON
                    last_error = exc
                    if attempt < args.max_retries:
                        time.sleep(min(10.0, 0.5 * (2**attempt)) + random.random())
            if not raw:
                error = repr(last_error)
        try:
            payload = json.loads(raw) if raw else {}
            activities, stats = validate(payload, source_by_id, context_by_id)
        except Exception as exc:
            activities = []
            stats = {"raw": 0, "not_list": 1, "bad_source_id": 0, "bad_decision": 0, "missing_phrase": 0, "valid": 0}
            error = error or repr(exc)
        diagnostic = {
            "source_file": job["path"].name,
            "chunk_id": job["chunk_id"],
            "pass": pass_name,
            "cache_tag": cache_tag,
            "clauses": len(records),
            "validation": stats,
            "error": error,
            "attempts": attempts,
            "cached": cached,
        }
        return job, activities, diagnostic

    def execute(items: list[dict[str, Any]], repair: bool = False) -> list[tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]]:
        results: list[tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]] = []
        for start in range(0, len(items), args.submission_batch_size):
            wave = items[start : start + args.submission_batch_size]
            with ThreadPoolExecutor(max_workers=min(args.api_workers, len(wave))) as pool:
                futures = [pool.submit(run, item, repair) for item in wave]
                for future in as_completed(futures):
                    results.append(future.result())
            print(f"{('Repair' if repair else 'Primary')} completed {min(start + len(wave), len(items))}/{len(items)}", flush=True)
        return results

    primary = execute(jobs)
    primary_by_key = {(item["path"].name, item["chunk_id"]): (activities, diag) for item, activities, diag in primary}
    repair_jobs = [
        job
        for job in jobs
        if primary_by_key[(job["path"].name, job["chunk_id"])][1]["validation"]["valid"] == 0
        or not response_is_structurally_valid(
            primary_by_key[(job["path"].name, job["chunk_id"])][1]
        )
    ]
    repairs = execute(repair_jobs, True) if repair_jobs else []
    selected = {
        (item["path"].name, item["chunk_id"]): (activities, diagnostic)
        for item, activities, diagnostic in primary
    }
    recovery_diagnostics: list[dict[str, Any]] = []
    transport_fallback_rows = 0
    unresolved_keys: set[tuple[str, int]] = set()

    def recover_records(
        job: dict[str, Any], records: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        nonlocal transport_fallback_rows
        first, last = records[0]["source_id"], records[-1]["source_id"]
        cache_tag = f"__bisect_{first}_{last}"
        _, activities, diagnostic = run(
            job, True, records_override=records, cache_tag=cache_tag
        )
        recovery_diagnostics.append(diagnostic)
        if response_is_structurally_valid(diagnostic):
            return activities
        if len(records) > 1:
            middle = len(records) // 2
            return recover_records(job, records[:middle]) + recover_records(job, records[middle:])
        if args.fallback_to_source_on_failure:
            transport_fallback_rows += 1
            return [transport_fallback_activity(records[0])]
        unresolved_keys.add((job["path"].name, job["chunk_id"]))
        return []

    for item, activities, diagnostic in repairs:
        key = (item["path"].name, item["chunk_id"])
        if response_is_structurally_valid(diagnostic):
            # A schema-valid empty list is a substantive result and supersedes
            # an invalid primary response just as a non-empty repair does.
            selected[key] = (activities, diagnostic)
            continue
        if args.bisect_on_failure:
            records = records_for_chunk(item["chunk"], args.clause_threshold)
            if len(records) > 1:
                middle = len(records) // 2
                recovered = recover_records(item, records[:middle]) + recover_records(
                    item, records[middle:]
                )
            else:
                recovered = recover_records(item, records)
            selected[key] = (recovered, diagnostic)
        else:
            unresolved_keys.add(key)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        activities, _ = selected[(job["path"].name, job["chunk_id"])]
        for activity in activities:
            rows.append(
                {
                    **{key: job["metadata"][key] for key in ("firm_id", "ticker", "company", "year", "accession")},
                    "model": args.model,
                    "source_file": job["path"].name,
                    "chunk_id": job["chunk_id"],
                    **activity,
                }
            )
    fields = METADATA_FIELDS + CORE_FIELDS + [
        "prior_context", "resolved_actor_text", "actor_resolution_status",
        "transport_fallback", "transport_fallback_reason",
    ]
    write_csv(args.output_dir / "extraction_audit.csv", rows, fields)
    write_csv(args.output_dir / "keep.csv", [row for row in rows if row["decision"] == "keep"], fields)
    write_csv(args.output_dir / "reject.csv", [row for row in rows if row["decision"] == "reject"], fields)
    diagnostics = (
        [diag for _, _, diag in primary]
        + [diag for _, _, diag in repairs]
        + recovery_diagnostics
    )
    atomic_json(args.output_dir / "chunk_diagnostics.json", diagnostics)
    final_zero = sum(not selected[(job["path"].name, job["chunk_id"])][0] for job in jobs)
    atomic_json(
        args.output_dir / "extraction_manifest.json",
        {
            "model": args.model,
            "cache_optimized_prompt_format": args.cache_optimized_prompt_format,
            "filings": len(prepared),
            "jobs": len(jobs),
            "primary_zero_or_error": len(repair_jobs),
            "repairs": len(repairs),
            "final_empty_after_repair": final_zero,
            "transport_fallback_rows": transport_fallback_rows,
            "unresolved_jobs": len(unresolved_keys),
            "rows": len(rows),
            "keep_rows": sum(row["decision"] == "keep" for row in rows),
            "reject_rows": sum(row["decision"] == "reject" for row in rows),
        },
    )
    print(json.dumps(json.loads((args.output_dir / "extraction_manifest.json").read_text(encoding="utf-8")), ensure_ascii=False), flush=True)
    # Primary errors that were repaired are historical diagnostics, not final
    # failures. Only jobs still unresolved after repair/bisection fail the run.
    if unresolved_keys:
        raise SystemExit("Extraction has unresolved API/JSON errors; rerun resumes from response_cache")


if __name__ == "__main__":
    main()
