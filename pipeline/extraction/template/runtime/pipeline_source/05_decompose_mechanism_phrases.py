#!/usr/bin/env python3
"""Condense audited activity phrases into concise mechanism phrases with an LLM.

The script preserves every source column and adds auditable LLM output columns.
It uses only the Python standard library; provide a DEEPSEEK_API_KEY environment
variable before running it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("audited_keep.csv")


SYSTEM_PROMPT = """You are a precise research-data editor. You are cleaning
firm-level activities extracted from 10-K filings into concise English
mechanism phrases. A mechanism phrase states what the focal firm actually does
or provides. It is not a summary of the whole sentence, a benefit, an intent,
or a marketing claim.

For each input record, return one core_mechanism_phrase and, only when needed,
an atomic_mechanism_phrases list.  In the formal no-source mode, the only
evidence supplied is replaced_activity_phrase.  Never infer or reintroduce a
company, brand, person, location, product, supplier, acronym, or other named
entity that is not present in that phrase.

Rules for core_mechanism_phrase:
1. Use a short, action-oriented verb phrase; omit the subject (we, the company,
   our customers). Preserve the source language and do not invent facts.
2. Keep the concrete mechanism, but remove framing, intent, benefits, outcomes,
   predictions, praise, and unnecessary setting/detail. In particular, remove
   wording such as "offer a unique experience by", "reinforced our commitment
   to", "designed to", "which enables", "in order to", and "to improve".
3. If a lengthy list merely elaborates one named, higher-level mechanism, use
   that higher-level mechanism instead of retaining its implementation details.
   For example, door materials, locks, cameras, and control-room feeds are
   details of "implement additional security measures".
4. Conversely, retain a short list of materially different customer offerings
   or operating levers when collapsing them would lose the mechanism. Prefer a
   shared verb plus a compact list rather than a full sentence.
5. If there are two or more genuinely independent action families, put the
   compact actions in core_mechanism_phrase separated by semicolons, and list
   those same action families in atomic_mechanism_phrases. Do not split
   sub-items that are only examples or specifications of one action.
6. Keep placeholders such as [airport] only if needed to identify the action.
   Do not introduce company names, people, locations, numbers, or dates that
   are absent from the activity phrase. Do not replace an explicit action with
   an inferred synonym or umbrella category unless that category appears in the
   activity phrase. For example, keep "do not use paper tickets" rather than
   inferring "use ticketless travel".
7. Aim for 3-25 words. No ending period. Flag needs_human_review when the
   source is ambiguous, when a concise faithful phrase is not possible, or
   when you are uncertain whether actions are independent.

Examples:
- Input: "offer passengers a unique flying experience by providing new
  aircraft, leather seats, simple and low fares, free LiveTV at every seat,
  pre-assigned seating and reliable performance"
  Output core: "provide new aircraft, leather seats, simple and low fares,
  free LiveTV at every seat, and pre-assigned seating"
  Atomic list: [the same core phrase]
- Input: "reinforced our commitment to safety by implementing additional
  security measures, including reinforcing all cockpit doors with
  bullet-resistant Kevlar and multiple titanium deadbolt locks capable of
  being opened only from within the cockpit, and commencing installation of
  four cabin security cameras on each aircraft with a live feed to the cockpit
  crew"
  Output core: "implement additional security measures"
  Atomic list: ["implement additional security measures"]
- Input: "all of our travel is ticketless and we strongly emphasize bookings
  through our website"
  Output core: "use ticketless travel; emphasize bookings through our website"
  Atomic list: ["use ticketless travel", "emphasize bookings through our website"]
- Input: "adding additional frequencies on existing routes, connecting new
  city pairs among the destinations we already serve and entering new markets"
  Output core: "increase frequencies on existing routes; connect new city
  pairs; enter new markets"
  Atomic list: ["increase frequencies on existing routes", "connect new city
  pairs", "enter new markets"]
- Input: "provide continuous training for our crewmembers including technical
  training, a leadership program, regular training focused on safety, and
  frontline customer service training"
  Output core: "provide continuous technical, leadership, safety, and
  customer-service training"
  Atomic list: [the same core phrase]

Return valid JSON only, with this exact structure (all keys are required):
{"items":[{"row_id":1,"core_mechanism_phrase":"...",
"atomic_mechanism_phrases":["..."],"decision":"unchanged|condensed|split",
"confidence":"high|medium|low","needs_human_review":false,
"rationale":"..."}]}"""


RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "row_id": {"type": "integer"},
                    "core_mechanism_phrase": {"type": "string"},
                    "atomic_mechanism_phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["unchanged", "condensed", "split"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "needs_human_review": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "row_id",
                    "core_mechanism_phrase",
                    "atomic_mechanism_phrases",
                    "decision",
                    "confidence",
                    "needs_human_review",
                    "rationale",
                ],
            },
        }
    },
    "required": ["items"],
}


ADDED_COLUMNS = [
    "phrase_decomposition_row_id",
    "core_mechanism_phrase",
    "atomic_mechanism_phrases_json",
    "phrase_decomposition_decision",
    "phrase_decomposition_confidence",
    "phrase_decomposition_needs_human_review",
    "phrase_decomposition_rationale",
    "phrase_decomposition_model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <input stem>_core_mechanisms.csv)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="DeepSeek model name (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI-compatible base URL (default: https://api.deepseek.com)",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable holding the DeepSeek key (default: DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional industry-neutral system prompt file; defaults to the legacy embedded prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Rows per LLM request (default: 8)"
    )
    parser.add_argument(
        "--api-workers", type=int, default=1,
        help="Concurrent LLM requests; checkpointing remains serialized (default: 1)",
    )
    parser.add_argument(
        "--exclude-source-sentence",
        action="store_true",
        help="Decompose from replaced_activity_phrase only; omit source sentence and mechanism context.",
    )
    parser.add_argument(
        "--limit", type=int, help="Process only the first N rows; useful for pilots"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call the model and print results, but do not create a CSV",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess rows already present in an existing output CSV",
    )
    parser.add_argument(
        "--max-retries", type=int, default=4, help="Retries for transient API errors"
    )
    parser.add_argument(
        "--fallback-to-source-on-failure",
        action="store_true",
        help="Retry failed batches row by row, then preserve the raw phrase with a review flag.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.api_workers < 1:
        parser.error("--api-workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_core_mechanisms.csv")
    return args


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), list(reader)


def existing_results(path: Path) -> dict[int, dict[str, str]]:
    """Read a prior output so an interrupted job can resume safely."""
    if not path.exists():
        return {}
    _, rows = read_csv(path)
    results: dict[int, dict[str, str]] = {}
    for row in rows:
        value = row.get("phrase_decomposition_row_id", "")
        if value.isdigit() and row.get("core_mechanism_phrase", "").strip():
            results[int(value)] = row
    return results


def compact_context(text: str, limit: int = 700) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def make_user_input(
    batch: list[tuple[int, dict[str, str]]], include_source_sentence: bool = True
) -> str:
    records = []
    for row_id, row in batch:
        phrase = row.get("replaced_activity_phrase", "").strip()
        # The clean/replace stage uses the literal value "none" when it makes
        # no entity substitution.  That is not an activity phrase: preserve
        # the cleaned phrase as the no-source decomposition evidence instead.
        if not phrase or phrase.lower() == "none":
            phrase = row.get("cleaned_activity_phrase", "").strip() or row.get("activity_phrase", "").strip()
        record = {
            "row_id": row_id,
            "replaced_activity_phrase": compact_context(phrase),
        }
        if include_source_sentence:
            record["source_sentence"] = compact_context(row.get("source_sentence", ""))
            record["mechanism_context"] = compact_context(row.get("mechanism_text", ""), 350)
        records.append(record)
    return json.dumps({"records": records}, ensure_ascii=False)


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("DeepSeek API returned no choices")
    if choices[0].get("finish_reason") == "length":
        raise ValueError("DeepSeek response was truncated; lower --batch-size")
    message = choices[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek API returned empty message content")
    return content


def singleton_json_regex(row_id: int) -> str:
    value = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
    return (r'\{"items":\[\{"row_id":' + str(row_id) + r',"core_mechanism_phrase":' + value
            + r',"atomic_mechanism_phrases":\[' + value + r'(?:,' + value + r'){0,11}\],'
            + r'"decision":"(?:unchanged|condensed|split)","confidence":"(?:high|medium|low)",'
            + r'"needs_human_review":(?:true|false),"rationale":' + value + r'\}\]\}')


def call_model(
    api_key: str,
    model: str,
    base_url: str,
    batch: list[tuple[int, dict[str, str]]],
    max_retries: int,
    temperature: float,
    system_prompt: str,
    include_source_sentence: bool = True,
    compact_recovery: bool = False,
) -> list[dict[str, Any]]:
    user_input = make_user_input(batch, include_source_sentence)
    required_ids = [row_id for row_id, _ in batch]
    glm_id_note = (
        "\n\nReturn exactly one item for each INPUT RECORD. Preserve the supplied "
        f"row_id values exactly: {required_ids}. IDs in examples above are "
        "illustrations, not answers. Do not substitute an example ID."
    )
    messages = (
        [{"role": "user", "content": system_prompt + glm_id_note + "\n\nINPUT RECORDS:\n" + user_input}]
        if model == "parcc/GLM" else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    )
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "max_tokens": 3000,
    }
    if model == "parcc/GLM":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        item_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "row_id": {"type": "integer"},
                "core_mechanism_phrase": {"type": "string"},
                "atomic_mechanism_phrases": {"type": "array", "items": {"type": "string"}},
                "decision": {"type": "string", "enum": ["unchanged", "condensed", "split"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "needs_human_review": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["row_id", "core_mechanism_phrase", "atomic_mechanism_phrases",
                         "decision", "confidence", "needs_human_review", "rationale"],
        }
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "decomposition_items", "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "properties": {"items": {"type": "array", "items": item_schema,
                                                "minItems": len(batch), "maxItems": len(batch)}},
                       "required": ["items"]},
        }}
        if compact_recovery and len(batch) == 1:
            # Keep the research rules; remove only the JSON illustration that
            # this endpoint can copy verbatim instead of processing the input.
            instructions = system_prompt.split("Return valid JSON only", 1)[0]
            payload["messages"] = [{"role": "user", "content": instructions + glm_id_note
                + "\nReturn real grounded phrases and a brief rationale, never literal ellipses. "
                "Return the fields row_id, core_mechanism_phrase, atomic_mechanism_phrases, "
                "decision, confidence, needs_human_review, rationale in the constrained JSON."
                + "\n\nINPUT RECORDS:\n" + user_input}]
            payload.pop("response_format")
            payload["structured_outputs"] = {"regex": singleton_json_regex(batch[0][0])}
            payload["max_tokens"] = 1024
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as reply:
                body = json.loads(reply.read().decode("utf-8"))
            parsed = json.loads(response_text(body))
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("Structured response has no items list")
            return items
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            # HTTP 4xx errors other than 408/429 are generally not transient.
            status = getattr(exc, "code", None)
            retryable = status is None or status in {408, 409, 429} or status >= 500
            if attempt >= max_retries or not retryable:
                # Do not echo an API error body: providers can include credential
                # fragments or other sensitive request information in it.
                hint = (
                    " Check the configured DeepSeek key and model name."
                    if status in {401, 403, 404}
                    else ""
                )
                raise RuntimeError(f"LLM request failed: {exc}.{hint}") from exc
            wait_seconds = min(30, 2**attempt)
            print(
                f"Transient API error ({exc}); retrying in {wait_seconds}s "
                f"[{attempt + 1}/{max_retries}]",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def validate_items(
    items: list[dict[str, Any]], expected_ids: set[int], allow_singleton_example_id: bool = False,
) -> dict[int, dict[str, Any]]:
    received: dict[int, dict[str, Any]] = {}
    required_fields = {
        "row_id",
        "core_mechanism_phrase",
        "atomic_mechanism_phrases",
        "decision",
        "confidence",
        "needs_human_review",
        "rationale",
    }
    for item in items:
        if set(item) != required_fields:
            raise ValueError(
                "Model result has an unexpected JSON shape: "
                f"expected {sorted(required_fields)}, got {sorted(item)}"
            )
        row_id = item.get("row_id")
        if (allow_singleton_example_id and len(items) == len(expected_ids) == 1
                and type(row_id) is int and row_id in {0, 1} and row_id not in expected_ids):
            actual_id = next(iter(expected_ids))
            print(f"Normalized illustrative singleton row_id {row_id} to {actual_id}", file=sys.stderr)
            item = dict(item, row_id=actual_id)
            row_id = actual_id
        phrase = item.get("core_mechanism_phrase")
        if not isinstance(row_id, int) or row_id not in expected_ids:
            raise ValueError(f"Unexpected row_id in model result: {row_id!r}")
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(f"Empty core mechanism phrase for row {row_id}")
        if phrase.strip() in {"..", "...", "…"}:
            raise ValueError(f"Placeholder core mechanism phrase for row {row_id}")
        atomic = item["atomic_mechanism_phrases"]
        if (
            not isinstance(atomic, list)
            or not atomic
            or not all(isinstance(part, str) and part.strip() for part in atomic)
        ):
            raise ValueError(f"Invalid atomic phrases for row {row_id}")
        if any(part.strip() in {"..", "...", "…"} for part in atomic):
            raise ValueError(f"Placeholder atomic phrase for row {row_id}")
        if item["decision"] not in {"unchanged", "condensed", "split"}:
            raise ValueError(f"Invalid decision for row {row_id}")
        if item["decision"] == "split" and len(atomic) < 2:
            # The atom count is authoritative; a one-atom response is not a
            # split even if GLM chose the wrong metadata label.
            item = dict(item)
            item["decision"] = (
                "unchanged" if atomic[0].strip().casefold() == phrase.strip().casefold()
                else "condensed"
            )
        if item["confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid confidence for row {row_id}")
        if not isinstance(item["needs_human_review"], bool):
            raise ValueError(f"Invalid review flag for row {row_id}")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError(f"Empty rationale for row {row_id}")
        if item["rationale"].strip() in {"..", "...", "…"}:
            raise ValueError(f"Placeholder rationale for row {row_id}")
        if row_id in received:
            raise ValueError(f"Duplicate result for row {row_id}")
        received[row_id] = item
    missing = expected_ids - received.keys()
    if missing:
        raise ValueError(f"Model did not return results for row IDs: {sorted(missing)}")
    return received


def reusable_results(path: Path, indexed_rows: list[tuple[int, dict[str, str]]], model: str) -> dict[int, dict[str, str]]:
    """Reuse only successful, validated decomposition with identical copied input."""
    if not path.is_file():
        return {}
    _, previous = read_csv(path)
    ids = [row.get("phrase_decomposition_row_id", "") for row in previous]
    if not all(ids) or len(set(ids)) != len(ids):
        return {}
    prior = {row["phrase_decomposition_row_id"]: row for row in previous}
    reusable = {}
    for row_id, source in indexed_rows:
        row = prior.get(str(row_id))
        if (not row or row.get("phrase_decomposition_model") != model
                or "Raw phrase preserved after decomposition failure" in row.get("phrase_decomposition_rationale", "")
                or any(row.get(key, "") != value for key, value in source.items())):
            continue
        try:
            review = row["phrase_decomposition_needs_human_review"].lower()
            if review not in {"true", "false"}:
                continue
            validate_items([{
                "row_id": row_id, "core_mechanism_phrase": row["core_mechanism_phrase"],
                "atomic_mechanism_phrases": json.loads(row["atomic_mechanism_phrases_json"]),
                "decision": row["phrase_decomposition_decision"],
                "confidence": row["phrase_decomposition_confidence"],
                "needs_human_review": review == "true", "rationale": row["phrase_decomposition_rationale"],
            }], {row_id})
        except (ValueError, KeyError, TypeError):
            continue
        reusable[row_id] = row
    return reusable


def annotate(row_id: int, row: dict[str, str], item: dict[str, Any], model: str) -> dict[str, str]:
    result = dict(row)
    result.update(
        {
            "phrase_decomposition_row_id": str(row_id),
            "core_mechanism_phrase": item["core_mechanism_phrase"].strip().rstrip("."),
            "atomic_mechanism_phrases_json": json.dumps(
                item["atomic_mechanism_phrases"], ensure_ascii=False
            ),
            "phrase_decomposition_decision": item["decision"],
            "phrase_decomposition_confidence": item["confidence"],
            "phrase_decomposition_needs_human_review": str(
                item["needs_human_review"]
            ).lower(),
            "phrase_decomposition_rationale": item["rationale"].strip(),
            "phrase_decomposition_model": model,
        }
    )
    return result


def fallback_item(row_id: int, row: dict[str, str], error: Exception) -> dict[str, Any]:
    phrase = row.get("replaced_activity_phrase", "").strip()
    if not phrase or phrase.lower() == "none":
        phrase = row.get("activity_phrase", "").strip()
    if not phrase:
        raise ValueError(f"Cannot preserve an empty source phrase for row {row_id}") from error
    return {
        "row_id": row_id,
        "core_mechanism_phrase": phrase,
        "atomic_mechanism_phrases": [phrase],
        "decision": "unchanged",
        "confidence": "low",
        "needs_human_review": True,
        "rationale": f"Raw phrase preserved after decomposition failure: {type(error).__name__}",
    }


def write_csv(path: Path, source_columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = source_columns + [column for column in ADDED_COLUMNS if column not in source_columns]
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # Dropbox's Windows sync process can briefly hold the destination between
    # frequent checkpoints. Retry the atomic swap instead of losing completed
    # API batches to a transient WinError 5.
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(2.0, 0.1 * (attempt + 1)))


def main() -> int:
    args = parse_args()
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        if not args.system_prompt_file.is_file():
            print(f"System prompt file does not exist: {args.system_prompt_file}", file=sys.stderr)
            return 2
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"{args.api_key_env} is required.", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        return 2

    source_columns, source_rows = read_csv(args.input)
    if "replaced_activity_phrase" not in source_columns:
        print("Input has no replaced_activity_phrase column.", file=sys.stderr)
        return 2

    indexed_rows = list(enumerate(source_rows, start=1))
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]
    prior = {} if args.no_resume else existing_results(args.output)
    if os.getenv("PHRASE_REPAIR_REUSE_OUTPUT", "") == "1":
        prior = reusable_results(args.output, indexed_rows, args.model)
        print(f"Reusing validated decomposition output for {len(prior)}/{len(indexed_rows)} rows.")
    pending = [(row_id, row) for row_id, row in indexed_rows if row_id not in prior]
    merged: dict[int, dict[str, str]] = {
        row_id: prior[row_id] for row_id, _ in indexed_rows if row_id in prior
    }

    print(
        f"Loaded {len(source_rows)} rows; processing {len(pending)} rows with {args.model}."
    )
    batches = [pending[start : start + args.batch_size] for start in range(0, len(pending), args.batch_size)]

    def request_checked(batch: list[tuple[int, dict[str, str]]]) -> dict[int, dict[str, Any]]:
        last_error: Exception | None = None
        compact_recovery = os.getenv("PHRASE_DECOMPOSE_COMPACT_RECOVERY", "") == "1"
        for validation_attempt in range(args.max_retries + 1):
            try:
                items = call_model(
                    api_key,
                    args.model,
                    args.base_url,
                    batch,
                    args.max_retries,
                    args.temperature,
                    system_prompt,
                    include_source_sentence=not args.exclude_source_sentence,
                    compact_recovery=compact_recovery,
                )
                checked = validate_items(items, {row_id for row_id, _ in batch},
                                         allow_singleton_example_id=args.model == "parcc/GLM")
                return checked
            except Exception as exc:
                last_error = exc
                if validation_attempt >= args.max_retries:
                    raise exc
                # call_model wraps response parsing/truncation ValueErrors in
                # RuntimeError after its bounded transport retries. Preserve
                # that cause so singleton recovery also handles those replies.
                response_invalid = isinstance(exc, ValueError) or isinstance(exc.__cause__, ValueError)
                if args.model == "parcc/GLM" and len(batch) == 1 and response_invalid:
                    compact_recovery = True
                print(
                    f"Invalid structured response ({exc}); retrying batch "
                    f"[{validation_attempt + 1}/{args.max_retries}]",
                    file=sys.stderr,
                )
        assert last_error is not None
        raise last_error

    def process_batch(batch: list[tuple[int, dict[str, str]]]) -> tuple[list[tuple[int, dict[str, str]]], dict[int, dict[str, Any]]]:
        try:
            return batch, request_checked(batch)
        except Exception as batch_error:
            if not args.fallback_to_source_on_failure:
                raise
            print(f"Batch failed; retrying {len(batch)} rows individually: {batch_error}", file=sys.stderr)
            checked: dict[int, dict[str, Any]] = {}
            for row_id, row in batch:
                try:
                    checked.update(request_checked([(row_id, row)]))
                except Exception as row_error:
                    print(f"Preserving row {row_id} after decomposition failure: {row_error}", file=sys.stderr)
                    checked[row_id] = fallback_item(row_id, row, row_error)
            return batch, checked

    with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        futures = [pool.submit(process_batch, batch) for batch in batches]
        for completed, future in enumerate(as_completed(futures), start=1):
            batch, checked = future.result()
            print(
                f"Completed batch {completed}/{len(batches)}: rows {batch[0][0]}-{batch[-1][0]}",
                file=sys.stderr,
            )
            for row_id, row in batch:
                merged[row_id] = annotate(row_id, row, checked[row_id], args.model)
            # Checkpoint in small groups: still resumable while avoiding a
            # tight create/replace loop that competes with Dropbox sync.
            if not args.dry_run and (completed % 5 == 0 or completed == len(batches)):
                output_rows = [merged[row_id] for row_id, _ in indexed_rows if row_id in merged]
                write_csv(args.output, source_columns, output_rows)

    output_rows = [merged[row_id] for row_id, _ in indexed_rows]
    if args.dry_run:
        for row in output_rows:
            print(
                json.dumps(
                    {
                        "row_id": row["phrase_decomposition_row_id"],
                        "source": row["replaced_activity_phrase"],
                        "core": row["core_mechanism_phrase"],
                        "atomic": json.loads(row["atomic_mechanism_phrases_json"]),
                        "review": row["phrase_decomposition_needs_human_review"],
                    },
                    ensure_ascii=False,
                )
            )
    else:
        print(f"Wrote {len(output_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
