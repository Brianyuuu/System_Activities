#!/usr/bin/env python3
"""Audit focal-firm entities contextually, replace them, and emit a generated map."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI


PLACEHOLDER_BY_RELATION = {
    "focal_firm": "[focal_firm]",
    "focal_airline": "[focal_airline]",
    "controlled_airline": "[controlled_airline]",
    "controlled_business": "[controlled_business]",
    "codeshare_partner": "[codeshare_partner]",
    "alliance_partner": "[alliance_partner]",
    "joint_venture_partner": "[joint_venture_partner]",
    "marketing_partner": "[marketing_partner]",
    "collaborator": "[collaborator]",
    "supplier": "[supplier]",
    "fuel_supplier": "[fuel_supplier]",
    "technology_provider": "[technology_provider]",
    "financial_partner": "[financial_partner]",
    "aircraft_lessor": "[aircraft_lessor]",
    "aircraft_lessee": "[aircraft_lessee]",
    "customer": "[customer]",
    "competitor": "[competitor]",
    "regulator": "[regulator]",
    "alliance": "[airline_alliance]",
    "airport": "[airport]",
    "city": "[city]",
    "country": "[country]",
    "region": "[region]",
    "external_organization": "[external_organization]",
}
ENTITY_TYPES = {
    "company", "airline", "brand", "alliance", "airport", "city",
    "country", "region", "regulator", "organization",
}
RELATION_ALIASES = {
    "focal_company": "focal_firm",
    "subsidiary": "controlled_business",
    "airline_partner": "collaborator",
    "partner": "collaborator",
    "vendor": "supplier",
    "government_agency": "regulator",
    "airport_code": "airport",
    "location": "region",
    "geography": "region",
}


def canonical_entity_type(entity_type: str, relation: str) -> str:
    if entity_type in ENTITY_TYPES:
        return entity_type
    if relation in {"airport", "city", "country", "region", "regulator", "alliance"}:
        return relation
    if relation in {"focal_airline", "controlled_airline", "codeshare_partner", "alliance_partner"}:
        return "airline"
    if relation in {"focal_firm", "controlled_business"}:
        return "company"
    return "organization"


def phrase_for_replacement(row: dict[str, str], preferred_field: str = "") -> str:
    """Use the audited atom when present; retain legacy core compatibility."""
    if preferred_field:
        return str(row.get(preferred_field) or "")
    return str(row.get("atom_mechanism_phrase") or row.get("core_mechanism_phrase") or "")


def named_entity_candidate_hints(text: str) -> list[str]:
    """High-recall surface hints only; the model still decides eligibility/role."""
    without_placeholders = re.sub(r"\[[a-z_]+\]", " ", text)
    pattern = r"(?<![\w])(?:[A-Z](?:[A-Za-z'’]*|(?:\.[A-Z])+\.?)|[A-Z]{2,})(?:[ /-](?:[A-Z](?:[A-Za-z'’]*|(?:\.[A-Z])+\.?)|[A-Z]{2,}))*"
    values: list[str] = []
    for match in re.finditer(pattern, without_placeholders):
        value = match.group(0).strip(" -/")
        if value and value not in values:
            values.append(value)
        for part in re.split(r"[-/]", value):
            part = part.strip()
            if part and part not in values:
                values.append(part)
    return values


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_invalid_response(
    cache_dir: Path, request_hash: str, attempt: int,
    expected_ids: set[int], response: Any, error: Exception,
) -> None:
    """Keep public model output and validation details, never request headers."""
    record: dict[str, Any] = {
        "request_hash": request_hash, "attempt": attempt,
        "expected_ids": sorted(expected_ids), "error_type": type(error).__name__,
    }
    if isinstance(error, ValueError):
        record["validation_error"] = str(error)
    if isinstance(error, json.JSONDecodeError):
        record["json_error_offset"] = error.pos
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        record["http_status"] = status_code
    if response is not None and getattr(response, "choices", None):
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        reasoning = (getattr(message, "model_extra", None) or {}).get("reasoning_content")
        record.update({"finish_reason": getattr(choice, "finish_reason", None),
                       "content": content, "reasoning_content": reasoning})
        usage = getattr(response, "usage", None)
        if usage is not None:
            record["usage"] = {key: getattr(usage, key, None) for key in
                               ("prompt_tokens", "completion_tokens", "total_tokens")}
        text = (content or "") + (reasoning or "")
        record["longest_whitespace_run"] = max(
            (len(match.group()) for match in re.finditer(r"\s+", text)), default=0,
        )
    try:
        atomic_json(cache_dir / "invalid_responses" /
                    f"{request_hash}_{attempt}_{time.time_ns()}.json", record)
    except OSError:
        print("Could not save entity diagnostic (OSError)", flush=True)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(payload: Any, expected_ids: set[int], phrases: dict[int, str]) -> dict[int, list[dict[str, str]]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Response lacks items list")
    result: dict[int, list[dict[str, str]]] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "replacements"}:
            raise ValueError("Each item must contain exactly id and replacements")
        item_id = item["id"]
        replacements = item["replacements"]
        if type(item_id) is not int or item_id not in expected_ids or item_id in result:
            raise ValueError("Unexpected or duplicate item id")
        if not isinstance(replacements, list):
            raise ValueError("replacements must be a list")
        checked: list[dict[str, str]] = []
        for replacement in replacements:
            required = {
                "entity_text", "entity_type", "relation_to_focal",
                "confidence", "evidence", "reason",
            }
            if not isinstance(replacement, dict) or set(replacement) != required:
                raise ValueError("Replacement has an invalid schema")
            entity = replacement["entity_text"]
            entity_type = replacement["entity_type"].strip().casefold()
            relation = replacement["relation_to_focal"].strip().casefold()
            relation = RELATION_ALIASES.get(relation, relation)
            confidence = replacement["confidence"]
            if not all(isinstance(value, str) and value.strip() for value in replacement.values()):
                raise ValueError("Replacement fields must be non-empty strings")
            if relation not in PLACEHOLDER_BY_RELATION:
                # Preserve strict placeholder control without failing a whole
                # batch when the model emits an unlisted role synonym. Place
                # types can be inferred safely from entity_type; other unknown
                # organizations use the conservative external role.
                relation = (
                    entity_type
                    if entity_type in {"airport", "city", "country", "region", "regulator", "alliance"}
                    else "external_organization"
                )
            if confidence not in {"high", "medium", "low"}:
                raise ValueError("Invalid confidence")
            # A model can occasionally identify a valid focal name from the
            # source sentence even though that span is absent from the final
            # core phrase.  It is not applicable to this deterministic rewrite
            # and is safely discarded rather than failing the whole batch.
            if entity.casefold() not in phrases[item_id].casefold():
                continue
            normalized = {key: str(replacement[key]).strip() for key in required}
            normalized["entity_type"] = canonical_entity_type(entity_type, relation)
            normalized["relation_to_focal"] = relation
            normalized["replacement"] = PLACEHOLDER_BY_RELATION[relation]
            checked.append(normalized)
        result[item_id] = checked
    if set(result) != expected_ids:
        raise ValueError("Response omitted one or more ids")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--firm", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument(
        "--bisect-on-failure",
        action="store_true",
        help="Split a failed multi-row request into smaller requests before retrying.",
    )
    parser.add_argument(
        "--fallback-to-no-replacements-on-failure",
        action="store_true",
        help="Keep a failed singleton phrase unchanged and mark it for audit.",
    )
    parser.add_argument(
        "--phrase-field",
        default="",
        help="Explicit input phrase column; default prefers atom_mechanism_phrase then core_mechanism_phrase.",
    )
    parser.add_argument(
        "--candidate-hints",
        action="store_true",
        help="Include deterministic proper-name/acronym surface candidates for high-recall residual auditing.",
    )
    args = parser.parse_args()

    if args.request_timeout <= 0 or args.max_tokens < 1:
        raise SystemExit("request-timeout and max-tokens must be positive")

    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields, rows = list(reader.fieldnames or []), list(reader)
    if not rows:
        raise SystemExit("Input CSV has no rows")
    template = args.prompt.read_text(encoding="utf-8")
    client = OpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=args.request_timeout,
    )
    indexed = list(enumerate(rows))
    batches = [indexed[start:start + args.batch_size] for start in range(0, len(indexed), args.batch_size)]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    fallback_rows: set[int] = set()
    fallback_lock = Lock()

    def work(
        batch: list[tuple[int, dict[str, str]]],
        retry_limit: int,
    ) -> dict[int, list[dict[str, str]]]:
        payload = []
        for index, row in batch:
            phrase = phrase_for_replacement(row, args.phrase_field)
            item = {
                "id": index,
                # Historical media lineages can contain two registrant names
                # under one stable pipeline firm ID. Preserve the row-level
                # filing company when available; legacy airline inputs without
                # that column retain the exact --firm behavior.
                "firm": row.get("company", "").strip() or args.firm,
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "atom_mechanism_phrase": phrase,
                "source_sentence": row.get("source_sentence", ""),
                "actor_text": row.get("actor_text", ""),
                "curated_actor": row.get("curated_actor", ""),
                "actor_relation_to_focal": row.get("actor_relation_to_focal", ""),
                "curated_actor_relation": row.get("curated_actor_relation", ""),
                "relationship_evidence": row.get("relationship_evidence", ""),
            }
            if args.candidate_hints:
                item["residual_candidate_spans"] = named_entity_candidate_hints(phrase)
            payload.append(item)
        prompt_hash = hashlib.sha256(
            json.dumps({"model": args.model, "prompt": template, "items": payload,
                        "prompt_format": "single_user_schema_v1" if args.model == "parcc/GLM" else "system_user"}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_file = args.cache_dir / f"{prompt_hash}.json"
        expected = {index for index, _ in batch}
        phrases = {index: phrase_for_replacement(row, args.phrase_field) for index, row in batch}
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("request_hash") == prompt_hash:
                return validate(cached.get("response"), expected, phrases)
        last_error: Exception | None = None
        for attempt in range(retry_limit + 1):
            response = None
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    response_format=(
                        {"type": "json_schema", "json_schema": {
                            "name": "entity_replacements", "strict": True,
                            "schema": {"type": "object", "additionalProperties": False,
                                       "properties": {"items": {"type": "array", "minItems": len(expected), "maxItems": len(expected), "items": {
                                           "type": "object", "additionalProperties": False,
                                           "properties": {
                                               "id": {"type": "integer", "enum": sorted(expected)},
                                               "replacements": {"type": "array", "items": {
                                                   "type": "object", "additionalProperties": False,
                                                   "properties": {key: {"type": "string"} for key in (
                                                       "entity_text", "entity_type", "relation_to_focal",
                                                       "confidence", "evidence", "reason")},
                                                   "required": ["entity_text", "entity_type", "relation_to_focal",
                                                                "confidence", "evidence", "reason"],
                                               }},
                                           }, "required": ["id", "replacements"],
                                       }}}, "required": ["items"]},
                        }} if args.model == "parcc/GLM" else {"type": "json_object"}
                    ),
                    messages=(
                        [{"role": "user", "content": template + "\n\nINPUT ITEMS:\n" + json.dumps({"items": payload}, ensure_ascii=False)}]
                        if args.model == "parcc/GLM" else [
                            {"role": "system", "content": template},
                            {"role": "user", "content": json.dumps({"items": payload}, ensure_ascii=False)},
                        ]
                    ),
                    extra_body={
                        "thinking": {"type": "disabled"},
                        **({"chat_template_kwargs": {"enable_thinking": False}} if args.model == "parcc/GLM" else {}),
                    },
                )
                message = response.choices[0].message
                parsed = json.loads(message.content or (message.model_extra or {}).get("reasoning_content") or "{}")
                validated = validate(parsed, expected, phrases)
                atomic_json(cache_file, {"request_hash": prompt_hash, "response": parsed})
                return validated
            except Exception as error:
                last_error = error
                save_invalid_response(args.cache_dir, prompt_hash, attempt,
                                      expected, response, error)
                if attempt < retry_limit:
                    time.sleep(min(10.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Contextual entity audit failed: {last_error}")

    def work_with_recovery(
        batch: list[tuple[int, dict[str, str]]],
    ) -> dict[int, list[dict[str, str]]]:
        retry_limit = 0 if args.bisect_on_failure and len(batch) > 1 else args.max_retries
        try:
            return work(batch, retry_limit)
        except Exception as error:
            if len(batch) == 1 and args.fallback_to_no_replacements_on_failure:
                index, _ = batch[0]
                with fallback_lock:
                    fallback_rows.add(index)
                print(
                    f"Contextual entity row {index + 1} failed after validated retries "
                    f"({type(error).__name__}); preserving phrase unchanged",
                    flush=True,
                )
                return {index: []}
            if not args.bisect_on_failure or len(batch) == 1:
                raise
            midpoint = len(batch) // 2
            left, right = batch[:midpoint], batch[midpoint:]
            first_id, last_id = batch[0][0], batch[-1][0]
            print(
                f"Contextual entity request for rows {first_id + 1}-{last_id + 1} "
                f"failed ({type(error).__name__}); retrying as "
                f"{len(left)}+{len(right)} rows",
                flush=True,
            )
            recovered: dict[int, list[dict[str, str]]] = {}
            for smaller_batch in (left, right):
                recovered.update(work_with_recovery(smaller_batch))
            return recovered

    results: dict[int, list[dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(work_with_recovery, batch) for batch in batches]
        for number, future in enumerate(as_completed(futures), 1):
            results.update(future.result())
            print(f"Contextual entity batches: {number}/{len(batches)}", flush=True)
    if len(results) != len(rows):
        raise RuntimeError("Contextual entity audit result count mismatch")

    added_fields = [
        "raw_atom_mechanism_phrase",
        "contextual_replacement_json",
        "final_mechanism_phrase",
        "final_replacement_status",
        "final_replacement_confidence",
        "final_replacement_reason",
    ]
    output_rows: list[dict[str, Any]] = []
    map_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        raw_phrase = phrase_for_replacement(row, args.phrase_field)
        if not raw_phrase.strip():
            raise ValueError(f"Row {index} lacks an atom/core mechanism phrase")
        text = raw_phrase
        applied: list[dict[str, str]] = []
        for replacement in sorted(results[index], key=lambda item: len(item["entity_text"]), reverse=True):
            # Low-confidence classifications remain visible in the audit JSON
            # but are not applied automatically.
            if replacement["confidence"] == "low":
                continue
            pattern = rf"(?<![\w]){re.escape(replacement['entity_text'])}(?![\w])"
            if not re.search(pattern, text, re.I):
                # A longer approved alias in the same phrase may already have
                # consumed this shorter overlapping span.
                continue
            text = re.sub(pattern, replacement["replacement"], text, flags=re.I)
            applied.append(replacement)
            key_tuple = (
                row.get("ticker", ""), row.get("year", ""),
                replacement["entity_text"], replacement["relation_to_focal"],
            )
            group = map_groups.setdefault(
                key_tuple,
                {
                    "firm_ticker": row.get("ticker", ""),
                    "year": row.get("year", ""),
                    "entity_text": replacement["entity_text"],
                    "entity_type": replacement["entity_type"],
                    "relation_to_focal": replacement["relation_to_focal"],
                    "placeholder": replacement["replacement"],
                    "occurrences": 0,
                    "status": "approved",
                    "confidence": replacement["confidence"],
                    "example_phrase": raw_phrase,
                    "source_sentence": row.get("source_sentence", ""),
                    "evidence": replacement["evidence"],
                    "reason": replacement["reason"],
                    "model": args.model,
                },
            )
            group["occurrences"] += 1
        output_rows.append(
            {
                **row,
                "raw_atom_mechanism_phrase": raw_phrase,
                "contextual_replacement_json": json.dumps(results[index], ensure_ascii=False),
                "final_mechanism_phrase": text,
                "final_replacement_status": (
                    "transport_fallback_unchanged"
                    if index in fallback_rows
                    else "rewritten" if applied else "unchanged"
                ),
                "final_replacement_confidence": (
                    "low"
                    if index in fallback_rows
                    else min(
                        (item["confidence"] for item in applied),
                        default="",
                        key={"low": 0, "medium": 1, "high": 2}.get,
                    )
                ),
                "final_replacement_reason": (
                    "Entity replacement API failed after validated retries; phrase preserved unchanged"
                    if index in fallback_rows
                    else "; ".join(item["reason"] for item in applied)
                ),
            }
        )

    write_csv(args.output, source_fields + [field for field in added_fields if field not in source_fields], output_rows)
    map_fields = [
        "firm_ticker", "year", "entity_text", "entity_type",
        "relation_to_focal", "placeholder", "occurrences", "status",
        "confidence", "example_phrase", "source_sentence", "evidence",
        "reason", "model",
    ]
    write_csv(
        args.map_output,
        map_fields,
        sorted(map_groups.values(), key=lambda row: (
            row["firm_ticker"], row["year"], row["entity_text"].casefold(),
            row["relation_to_focal"],
        )),
    )
    manifest = {
        "input": str(args.input),
        "rows": len(rows),
        "rewritten_rows": sum(row["final_replacement_status"] == "rewritten" for row in output_rows),
        "generated_map_rows": len(map_groups),
        "transport_fallback_rows": len(fallback_rows),
        "firm": args.firm,
        "model": args.model,
    }
    atomic_json(args.output.parent / "contextual_entity_replacement_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
