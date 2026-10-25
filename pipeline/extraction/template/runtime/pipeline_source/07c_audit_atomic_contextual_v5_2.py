#!/usr/bin/env python3
"""Audit core mechanism atoms with source and parent-phrase context.

The model only evaluates evidence flags.  This script derives the final audit pass/fail
decision deterministically, caches every validated response, and writes row-level audit
tables suitable for review or later rule changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI
import openai._base_client as openai_base_client


openai_base_client.get_platform = lambda: "Windows"


BOOL_FIELDS = (
    "source_supports_phrase",
    "focal_firm_explicit",
    "concrete_action",
    "mechanism_present",
    "repeatable_or_durable",
    "one_off_or_timed_event",
    "static_fact_or_feature",
    "customer_or_external_actor",
    "ordinary_support_or_hr",
    "operating_scale_or_status_fact",
)
CONTROLLED_SUBSIDIARY_CLASSES = {"focal_subsidiary_or_controlled_unit", "controlled_subsidiary"}
ASSET_OPERATION = re.compile(r"\b(?:operate|operates|operated|operating|operation(?:s)?\s+of)\b", re.I)
STRATEGIC_ASSET = re.compile(r"\b(?:refinery|pipeline|terminal(?:\s+assets?)?|plant|factory|warehouse)\b", re.I)
BUSINESS_ACTION = re.compile(
    r"\b(?:operate|operates|operating|provide|provides|providing|maintain|maintains|maintaining|"
    r"purchase|purchases|purchasing|procure|procures|procuring|supply|supplies|supplying|"
    r"manage|manages|managing|repair|repairs|repairing|overhaul|overhauls|service|services|"
    r"administer|administers|award|awards|redeem|redeems|contract|contracts|sign|signs)\b",
    re.I,
)
SPECIFIC_BUSINESS_CHOICE = re.compile(
    r"\b(?:under (?:our|the) code|capacity[- ]purchase|contract|agreement|offtake|supplier|"
    r"maintenance and engineering|maintenance support|aircraft charters?[,; ]+(?:aircraft )?management|"
    r"programs? allowing members|earn miles|redeem miles|award travel|ownership|lease|outsourc|"
    r"refinery|pipeline|terminal|fleet standardi[sz]|hub[- ]and[- ]spoke|distribution channel)\b",
    re.I,
)
SCALE_OR_SERVICE_DETAIL = re.compile(
    r"\b(?:\d+|number of|daily|weekly|nonstop|frequency|flights?|destinations?|cities|aircraft|"
    r"routes?|markets?|airports?|seats?|capacity|utili[sz]ation|schedule)\b",
    re.I,
)
GENERIC_MANAGEMENT = re.compile(
    r"\b(?:monitor|evaluate|assess|analy[sz]e|consider|plan|intend|seek|aim|expect|discuss)\b",
    re.I,
)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record.get("status") == "ok" and record.get("request_hash"):
                result[str(record["request_hash"])] = record
        except json.JSONDecodeError:
            continue
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate(payload: dict[str, Any], expected_ids: set[int]) -> list[dict[str, Any]]:
    audits = payload.get("audits") if isinstance(payload, dict) else None
    if not isinstance(audits, list):
        raise ValueError("Response lacks an audits list")
    seen: set[int] = set()
    output: list[dict[str, Any]] = []
    for item in audits:
        if not isinstance(item, dict) or set(item) != {"candidate_id", *BOOL_FIELDS}:
            raise ValueError("Response item has missing or additional fields")
        candidate_id = item.get("candidate_id")
        if type(candidate_id) is not int or candidate_id not in expected_ids or candidate_id in seen:
            raise ValueError("Unexpected or duplicate candidate_id")
        if any(type(item[field]) is not bool for field in BOOL_FIELDS):
            raise ValueError("Every audit field must be a JSON boolean")
        seen.add(candidate_id)
        output.append(item)
    if seen != expected_ids:
        raise ValueError("Response omitted one or more candidate_ids")
    return output


def derived_result(audit: dict[str, Any]) -> tuple[bool, str]:
    failures = []
    for field in ("source_supports_phrase", "focal_firm_explicit", "concrete_action", "mechanism_present", "repeatable_or_durable"):
        if not audit[field]:
            failures.append(field)
    for field in ("one_off_or_timed_event", "static_fact_or_feature", "customer_or_external_actor", "ordinary_support_or_hr", "operating_scale_or_status_fact"):
        if audit[field]:
            failures.append(field)
    return not failures, "|".join(failures) if failures else "pass"


def is_controlled_asset_operation(row: dict[str, str]) -> bool:
    actor_class = row.get("actor_class", "").strip().lower()
    relationship = row.get("ownership_relationship", "").strip().lower()
    focal = row.get("is_focal_firm_activity", "").strip().lower() in {"yes", "true", "1"}
    controlled = actor_class in CONTROLLED_SUBSIDIARY_CLASSES or "controlled" in relationship or "wholly_owned" in relationship
    phrase = row.get("replaced_activity_phrase", row.get("activity_phrase", ""))
    source = row.get("source_sentence", "")
    return bool(
        focal
        and controlled
        and ASSET_OPERATION.search(phrase)
        and STRATEGIC_ASSET.search(phrase)
        and ASSET_OPERATION.search(source)
        and STRATEGIC_ASSET.search(source)
    )


def controlled_asset_audit() -> dict[str, bool]:
    return {
        "source_supports_phrase": True,
        "focal_firm_explicit": True,
        "concrete_action": True,
        "mechanism_present": True,
        "repeatable_or_durable": True,
        "one_off_or_timed_event": False,
        "static_fact_or_feature": False,
        "customer_or_external_actor": False,
        "ordinary_support_or_hr": False,
        "operating_scale_or_status_fact": False,
    }


def is_retained_business_activity(row: dict[str, str], audit: dict[str, Any]) -> bool:
    """Keep only specific operating-model choices, not generic operations."""
    actor_is_retained = (
        row.get("curated_actor_filter", "").strip().lower() == "keep"
        or row.get("is_focal_firm_activity", "").strip().lower() in {"yes", "true", "1"}
    )
    phrase = row.get("core_mechanism_phrase") or row.get("replaced_activity_phrase") or row.get("activity_phrase", "")
    source = row.get("source_sentence", "")
    text = f"{phrase} {source}"
    return bool(
        actor_is_retained
        and audit["source_supports_phrase"]
        and audit["repeatable_or_durable"]
        and BUSINESS_ACTION.search(text)
        and SPECIFIC_BUSINESS_CHOICE.search(text)
        and not GENERIC_MANAGEMENT.search(phrase)
        and not (SCALE_OR_SERVICE_DETAIL.search(phrase) and not SPECIFIC_BUSINESS_CHOICE.search(phrase))
    )


def retained_business_activity_audit(audit: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(audit)
    repaired.update({
        "focal_firm_explicit": True,
        "concrete_action": True,
        "mechanism_present": True,
        "static_fact_or_feature": False,
        "customer_or_external_actor": False,
        "operating_scale_or_status_fact": False,
    })
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--bisect-on-failure", action="store_true")
    parser.add_argument("--fallback-to-reject-on-failure", action="store_true")
    parser.add_argument(
        "--omit-upstream-context",
        action="store_true",
        help="Send only focal firm, candidate phrase, and source sentence; omit prior extraction labels.",
    )
    parser.add_argument(
        "--keep-controlled-asset-operations",
        action="store_true",
        help="Keep explicit ongoing operation of a strategic asset by a focal controlled subsidiary.",
    )
    args = parser.parse_args()

    if args.max_retries < 1 or args.request_timeout <= 0 or args.max_tokens < 1:
        raise SystemExit("max-retries, request-timeout, and max-tokens must be positive")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    rows = list(csv.DictReader(args.input_csv.open(encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit("Input CSV has no rows")
    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    if "{candidate_records_json}" not in prompt_template:
        raise SystemExit("Prompt must contain {candidate_records_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "audit_responses.jsonl"
    cache = load_cache(cache_path)

    indexed = list(enumerate(rows))
    batches = [indexed[start : start + args.batch_size] for start in range(0, len(indexed), args.batch_size)]

    fallback_rows: set[int] = set()
    fallback_lock = Lock()

    def run_batch(
        batch: list[tuple[int, dict[str, str]]],
        attempts: int,
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        records = []
        for candidate_id, row in batch:
            record = {
                "candidate_id": candidate_id,
                "focal_firm": row.get("company", row.get("ticker", "")),
                "core_mechanism_phrase": row.get("core_mechanism_phrase", row.get("replaced_activity_phrase", row.get("activity_phrase", ""))),
                "source_sentence": row.get("source_sentence", ""),
                "parent_activity_context": row.get("pre_split_replaced_activity_phrase", row.get("replaced_activity_phrase", "")),
            }
            if not args.omit_upstream_context:
                record.update({
                    "existing_decision": row.get("decision", ""),
                    "existing_actor_text": row.get("actor_text", ""),
                    "existing_actor_class": row.get("actor_class", ""),
                    "existing_is_focal_firm_activity": row.get("is_focal_firm_activity", ""),
                    "resolved_actor_text": row.get("resolved_actor_text", ""),
                    "curated_actor": row.get("curated_actor", ""),
                    "curated_actor_relation": row.get("curated_actor_relation", ""),
                    "curated_actor_filter": row.get("curated_actor_filter", ""),
                })
            records.append(record)
        prompt = prompt_template.replace("{candidate_records_json}", json.dumps(records, ensure_ascii=False))
        request_hash = sha256(json.dumps({"model": args.model, "temperature": args.temperature, "stage": "boolean_phrase_audit", "prompt": prompt}, ensure_ascii=False, sort_keys=True))
        expected = {candidate_id for candidate_id, _ in batch}
        record = cache.get(request_hash)
        if record:
            try:
                validated = validate(record["parsed_response"], expected)
                return {item["candidate_id"]: item for item in validated}, []
            except (KeyError, ValueError):
                pass
        last_error: Exception | None = None
        for attempt in range(attempts):
            retry_prompt = prompt if not attempt else prompt + "\n\nCORRECTION: Return only the exact Boolean JSON schema for every candidate_id."
            try:
                request_payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": retry_prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "thinking": {"type": "disabled"},
                }
                if args.model == "parcc/GLM":
                    request_payload["chat_template_kwargs"] = {"enable_thinking": False}
                    audit_properties = {"candidate_id": {"type": "integer"},
                                        **{field: {"type": "boolean"} for field in BOOL_FIELDS}}
                    request_payload["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": "atomic_audits", "strict": True,
                        "schema": {"type": "object", "additionalProperties": False,
                                   "properties": {"audits": {"type": "array", "items": {
                                       "type": "object", "additionalProperties": False,
                                       "properties": audit_properties,
                                       "required": list(audit_properties),
                                   }}}, "required": ["audits"]},
                    }}
                request = urllib.request.Request(
                    args.api_base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(request_payload).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=args.request_timeout) as reply:
                    response = json.loads(reply.read().decode("utf-8"))
                message = response["choices"][0]["message"]
                raw = message.get("content") or message.get("reasoning_content") or "{}"
                payload = json.loads(raw)
                validated = validate(payload, expected)
                saved = {
                    "status": "ok", "request_hash": request_hash, "stage": "boolean_phrase_audit",
                    "model": args.model, "candidate_ids": sorted(expected), "parsed_response": payload,
                    "raw_response": raw, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                return {item["candidate_id"]: item for item in validated}, [saved]
            except Exception as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(min(10.0, 0.5 * (2 ** attempt)))
        assert last_error is not None
        raise last_error

    def run_batch_with_recovery(
        batch: list[tuple[int, dict[str, str]]],
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        attempts = 1 if args.bisect_on_failure and len(batch) > 1 else args.max_retries
        try:
            return run_batch(batch, attempts)
        except Exception as error:
            if len(batch) == 1 and args.fallback_to_reject_on_failure:
                candidate_id, _ = batch[0]
                with fallback_lock:
                    fallback_rows.add(candidate_id)
                print(
                    f"Atomic audit row {candidate_id + 1} failed after validated retries "
                    f"({type(error).__name__}); conservatively rejecting",
                    flush=True,
                )
                audit = {
                    "candidate_id": candidate_id,
                    **{field: False for field in BOOL_FIELDS},
                }
                return {candidate_id: audit}, []
            if not args.bisect_on_failure or len(batch) == 1:
                raise
            midpoint = len(batch) // 2
            left, right = batch[:midpoint], batch[midpoint:]
            print(
                f"Atomic audit rows {batch[0][0] + 1}-{batch[-1][0] + 1} "
                f"failed ({type(error).__name__}); retrying as {len(left)}+{len(right)} rows",
                flush=True,
            )
            values: dict[int, dict[str, Any]] = {}
            records: list[dict[str, Any]] = []
            for smaller in (left, right):
                smaller_values, smaller_records = run_batch_with_recovery(smaller)
                values.update(smaller_values)
                records.extend(smaller_records)
            return values, records

    by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.api_workers)) as pool:
        futures = [pool.submit(run_batch_with_recovery, batch) for batch in batches]
        for number, future in enumerate(as_completed(futures), start=1):
            values, new_records = future.result()
            by_index.update(values)
            for new_record in new_records:
                append_jsonl(cache_path, new_record)
                cache[new_record["request_hash"]] = new_record
            print(f"Audited {number}/{len(batches)} batches", flush=True)
    if len(by_index) != len(rows):
        raise RuntimeError("Audit result count mismatch")

    audited: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        audit = by_index[index]
        override_applied = False
        activity_override_applied = False
        passed, failure_codes = derived_result(audit)
        audited.append({
            **row,
            **{field: audit[field] for field in BOOL_FIELDS},
            "controlled_asset_operation_override": "yes" if override_applied else "no",
            "controlled_asset_operation_override_reason": "explicit controlled-subsidiary operation of a strategic asset" if override_applied else "",
            "retained_business_activity_override": "yes" if activity_override_applied else "no",
            "retained_business_activity_override_reason": "source-supported recurring action by retained actor" if activity_override_applied else "",
            "audit_transport_fallback": index in fallback_rows,
            "audit_transport_fallback_reason": "Audit API failed after validated retries; conservatively rejected" if index in fallback_rows else "",
            "audit_pass": passed,
            "audit_failure_codes": failure_codes,
        })
    fields = list(audited[0])
    write_csv(args.output_dir / "phrase_audit.csv", audited, fields)
    write_csv(args.output_dir / "audited_keep.csv", [row for row in audited if row["audit_pass"]], fields)
    write_csv(args.output_dir / "audited_reject.csv", [row for row in audited if not row["audit_pass"]], fields)
    manifest = {
        "pipeline": "boolean_schema_atomic_contextual_audit_v5_2", "model": args.model,
        "input_csv": str(args.input_csv), "prompt_file": str(args.prompt_file),
        "input_rows": len(rows), "audit_pass_rows": sum(row["audit_pass"] for row in audited),
        "audit_reject_rows": sum(not row["audit_pass"] for row in audited),
        "response_cache_records": len(load_cache(cache_path)),
        "transport_fallback_reject_rows": len(fallback_rows),
        "omit_upstream_context": args.omit_upstream_context,
        "keep_controlled_asset_operations": args.keep_controlled_asset_operations,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
