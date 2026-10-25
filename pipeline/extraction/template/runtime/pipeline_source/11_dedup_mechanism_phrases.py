#!/usr/bin/env python3
"""Deduplicate atomic mechanism phrases with exact and audited semantic stages.

The input must contain one atomic mechanism per row (normally the output of
``core_mechanism_phrases.csv``).  Exact matches are deterministic.  Semantic
matches are only merged after an LLM returns ``duplicate`` for a candidate
pair; every considered pair and decision is written to a CSV audit file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EXACT_COLUMNS = ["mechanism_exact_group_id", "mechanism_exact_group_size", "mechanism_exact_dedup_decision"]
SEMANTIC_COLUMNS = ["mechanism_semantic_group_id", "mechanism_semantic_group_size", "mechanism_semantic_canonical_phrase", "mechanism_semantic_dedup_decision", "mechanism_semantic_dedup_model"]
MATERIAL_QUALIFIERS = {"logistics", "pipeline", "terminal", "warehouse", "plant", "factory"}
MATERIAL_ACTIONS = {"own", "lease", "outsource", "manage"}


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\[\]_ ]+", " ", value.lower())).strip()


def phrase_for_dedup(row: dict[str, Any]) -> str:
    """Prefer replacement text, then the audited atom, then legacy core text."""
    return str(
        row.get("final_mechanism_phrase")
        or row.get("atom_mechanism_phrase")
        or row["core_mechanism_phrase"]
    )


def firm_key(row: dict[str, Any]) -> str:
    """Return a stable firm identity for scoped deduplication."""
    return normalise(str(row.get("ticker") or row.get("firm_id") or row.get("company") or ""))


def has_material_difference(left_phrase: str, right_phrase: str) -> bool:
    left_tokens = set(re.findall(r"[a-z]+", left_phrase.lower()))
    right_tokens = set(re.findall(r"[a-z]+", right_phrase.lower()))
    for group in (MATERIAL_QUALIFIERS, MATERIAL_ACTIONS):
        if bool(left_tokens & group) != bool(right_tokens & group):
            return True
    return False


def token_jaccard(left: str, right: str) -> float:
    a, b = set(normalise(left).split()), set(normalise(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def mechanism_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    """Stable ordering for readable sequential IDs and reproducible outputs."""
    try:
        row_id = int(str(row.get("phrase_decomposition_row_id", "")))
    except ValueError:
        row_id = sys.maxsize
    return (str(row.get("year", "")), row_id, normalise(phrase_for_dedup(row)))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def validate_assessments(payload: dict[str, Any], expected: set[int]) -> dict[int, dict[str, str]]:
    items = payload.get("assessments") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Response lacks assessments list")
    seen: dict[int, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"pair_id", "label", "canonical_phrase", "reason"}:
            raise ValueError("Each assessment must have exactly pair_id, label, canonical_phrase, reason")
        pair_id, label = item.get("pair_id"), item.get("label")
        if type(pair_id) is not int or pair_id not in expected or pair_id in seen:
            raise ValueError("Unexpected or duplicate pair_id")
        if label not in {"duplicate", "related_but_distinct", "not_duplicate"}:
            raise ValueError("Invalid semantic label")
        if not isinstance(item.get("canonical_phrase"), str) or not isinstance(item.get("reason"), str):
            raise ValueError("canonical_phrase and reason must be strings")
        seen[pair_id] = {"label": label, "canonical_phrase": item["canonical_phrase"].strip(), "reason": item["reason"].strip()}
    if set(seen) != expected:
        raise ValueError("Response omitted pair IDs")
    return seen


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse strict JSON while tolerating code fences or harmless trailing text."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Semantic response is not a JSON object")
    return value


def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    records: list[dict[str, Any]],
    retries: int,
    temperature: float,
    request_timeout: float,
    max_tokens: int,
) -> dict[int, dict[str, str]]:
    expected = {record["pair_id"] for record in records}
    user_payload = json.dumps({"pairs": records}, ensure_ascii=False)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            correction = "" if attempt == 0 else (
                "\n\nCORRECTION: Return one concise assessment for every pair_id. "
                "Output only a complete JSON object with no markdown or analysis."
            )
            payload = {
                "model": model,
                "messages": (
                    [{"role": "user", "content": system_prompt + "\n\nINPUT PAIRS:\n" + user_payload + correction}]
                    if model == "parcc/GLM" else [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload + correction},
                    ]
                ),
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if model == "parcc/GLM":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "semantic_assessments", "strict": True,
                    "schema": {"type": "object", "additionalProperties": False,
                               "properties": {"assessments": {"type": "array", "items": {
                                   "type": "object", "additionalProperties": False,
                                   "properties": {
                                       "pair_id": {"type": "integer"},
                                       "label": {"type": "string", "enum": ["duplicate", "related_but_distinct", "not_duplicate"]},
                                       "canonical_phrase": {"type": "string"},
                                       "reason": {"type": "string"},
                                   },
                                   "required": ["pair_id", "label", "canonical_phrase", "reason"],
                               }}}, "required": ["assessments"]},
                }}
            else:
                payload["reasoning_effort"] = "low"
            request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=request_timeout) as reply:
                body = json.loads(reply.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content")
            return validate_assessments(parse_json_object(content), expected)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(min(10.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Semantic request failed after {retries + 1} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-prompt", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-workers", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sequence-threshold", type=float, default=0.72)
    parser.add_argument("--jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument(
        "--conservative-no-merge-on-failure",
        action="store_true",
        help="Keep both phrases when a semantic pair cannot be validated.",
    )
    parser.add_argument(
        "--within-year-only",
        action="store_true",
        help="Deduplicate only within filing year; retain all cross-year mechanisms.",
    )
    parser.add_argument(
        "--within-firm-year-only",
        action="store_true",
        help="Deduplicate only within the same firm and filing year; retain all cross-firm and cross-year mechanisms.",
    )
    parser.add_argument(
        "--preserve-material-qualifiers",
        action="store_true",
        help="Do not semantically merge phrases when logistics/assets or ownership/management actions differ.",
    )
    args = parser.parse_args()

    if args.request_timeout <= 0 or args.max_tokens < 1:
        raise SystemExit("request-timeout and max-tokens must be positive")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "core_mechanism_phrase" not in reader.fieldnames:
            raise SystemExit("Input must have a core_mechanism_phrase column")
        source_fields, rows = list(reader.fieldnames), list(reader)
    if not rows:
        raise SystemExit("Input has no rows")
    if any(not row["core_mechanism_phrase"].strip() for row in rows):
        raise SystemExit("Input has an empty core_mechanism_phrase")
    if args.within_year_only and args.within_firm_year_only:
        raise SystemExit("Choose only one of --within-year-only and --within-firm-year-only")
    if args.within_firm_year_only and any(
        not firm_key(row) or not row.get("year", "").strip() for row in rows
    ):
        raise SystemExit("Firm-year deduplication requires a non-empty firm identity and year on every row")

    # Stage 1: exact matching over the atomic mechanism.  In annual mode, year
    # is part of the key so an identical mechanism in another filing year survives.
    groups: dict[tuple[str, ...] | str, list[dict[str, str]]] = {}
    for row in rows:
        phrase_key = normalise(phrase_for_dedup(row))
        if args.within_firm_year_only:
            group_key: tuple[str, ...] | str = (firm_key(row), row.get("year", ""), phrase_key)
        elif args.within_year_only:
            group_key = (row.get("year", ""), phrase_key)
        else:
            group_key = phrase_key
        groups.setdefault(group_key, []).append(row)
    exact_audit: list[dict[str, Any]] = []
    exact_kept: list[dict[str, Any]] = []
    for number, members in enumerate(groups.values(), 1):
        members.sort(key=lambda row: (row.get("year", ""), row.get("phrase_decomposition_row_id", ""), row.get("source_sentence", "")))
        for position, row in enumerate(members):
            enriched = {**row, "mechanism_exact_group_id": f"MEXACT{number:05d}", "mechanism_exact_group_size": len(members), "mechanism_exact_dedup_decision": "kept" if position == 0 else "duplicate"}
            exact_audit.append(enriched)
            if position == 0:
                exact_kept.append(enriched)
    exact_kept.sort(key=lambda row: (row.get("year", ""), row.get("phrase_decomposition_row_id", "")))

    # Stage 2: lexical candidate recall followed by strict LLM semantic decisions.
    pairs: list[dict[str, Any]] = []
    material_qualifier_pair_exclusions = 0
    for left in range(len(exact_kept)):
        for right in range(left + 1, len(exact_kept)):
            if args.within_firm_year_only and (
                firm_key(exact_kept[left]) != firm_key(exact_kept[right])
                or exact_kept[left].get("year", "") != exact_kept[right].get("year", "")
            ):
                continue
            if args.within_year_only and exact_kept[left].get("year", "") != exact_kept[right].get("year", ""):
                continue
            left_phrase, right_phrase = phrase_for_dedup(exact_kept[left]), phrase_for_dedup(exact_kept[right])
            sequence = SequenceMatcher(None, normalise(left_phrase), normalise(right_phrase)).ratio()
            jaccard = token_jaccard(left_phrase, right_phrase)
            if sequence >= args.sequence_threshold or jaccard >= args.jaccard_threshold:
                if args.preserve_material_qualifiers and has_material_difference(left_phrase, right_phrase):
                    material_qualifier_pair_exclusions += 1
                    continue
                # Prompt examples are one-based; matching that convention avoids
                # models renumbering the first pair from 0 to 1.
                pairs.append({"pair_id": len(pairs) + 1, "left_index": left, "right_index": right, "left_phrase": left_phrase, "right_phrase": right_phrase, "left_source_sentence": exact_kept[left].get("source_sentence", ""), "right_source_sentence": exact_kept[right].get("source_sentence", ""), "left_year": exact_kept[left].get("year", ""), "right_year": exact_kept[right].get("year", ""), "sequence_similarity": round(sequence, 4), "token_jaccard": round(jaccard, 4)})

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required for semantic deduplication")
    prompt = args.semantic_prompt.read_text(encoding="utf-8")
    batches = [pairs[start:start + args.pair_batch_size] for start in range(0, len(pairs), args.pair_batch_size)]
    decisions: dict[int, dict[str, str]] = {}
    failures: list[str] = []
    conservative_no_merge_pairs = 0
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(batch: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        # The prompt numbers examples within a batch.  Give the model compact
        # one-based IDs, then map its decisions back to the globally stable
        # pair IDs used by the audit output.  This prevents a final partial
        # batch such as global pair 41 from being repeatedly renumbered to 1.
        local_to_global: dict[int, int] = {}
        records: list[dict[str, Any]] = []
        for local_id, row in enumerate(batch, start=1):
            local_to_global[local_id] = row["pair_id"]
            record = {
                key: value
                for key, value in row.items()
                if key not in {"pair_id", "left_index", "right_index"}
            }
            records.append({"pair_id": local_id, **record})
        request_hash = hashlib.sha256(
            json.dumps(
                {"model": args.model, "prompt": prompt, "records": records,
                 "prompt_format": "single_user_schema_v1" if args.model == "parcc/GLM" else "system_user"},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_file = args.cache_dir / f"{request_hash}.json" if args.cache_dir else None
        if cache_file and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_items = cached.get("assessments") if cached.get("request_hash") == request_hash else None
            if isinstance(cached_items, list):
                local_decisions = validate_assessments(
                    {"assessments": cached_items}, set(local_to_global)
                )
                return {
                    local_to_global[local_id]: decision
                    for local_id, decision in local_decisions.items()
                }
        local_decisions = call_llm(
            api_key,
            args.base_url,
            args.model,
            prompt,
            records,
            args.max_retries,
            args.temperature,
            args.request_timeout,
            args.max_tokens,
        )
        if cache_file:
            atomic_json(
                cache_file,
                {
                    "request_hash": request_hash,
                    "assessments": [
                        {"pair_id": local_id, **local_decisions[local_id]}
                        for local_id in sorted(local_decisions)
                    ],
                },
            )
        return {
            local_to_global[local_id]: decision
            for local_id, decision in local_decisions.items()
        }
    with ThreadPoolExecutor(max_workers=max(1, args.api_workers)) as pool:
        future_map = {pool.submit(run_batch, batch): batch for batch in batches}
        for number, future in enumerate(as_completed(future_map), 1):
            batch = future_map[future]
            try:
                decisions.update(future.result())
            except Exception as error:
                pair_ids = [row["pair_id"] for row in batch]
                if args.conservative_no_merge_on_failure:
                    for row in batch:
                        decisions[row["pair_id"]] = {
                            "label": "not_duplicate",
                            "canonical_phrase": "",
                            "reason": (
                                "Semantic deduplication API failed after validated retries; "
                                "conservative no-merge fallback preserved both phrases"
                            ),
                        }
                    conservative_no_merge_pairs += len(batch)
                    print(
                        f"Semantic pair(s) {pair_ids} failed ({type(error).__name__}); "
                        "keeping both phrases",
                        flush=True,
                    )
                else:
                    failures.append(f"pair_ids={pair_ids}: {error}")
            print(f"Semantic batches: {number}/{len(batches)}", flush=True)
    if failures or len(decisions) != len(pairs):
        raise RuntimeError(f"Semantic dedup incomplete: {len(decisions)}/{len(pairs)} pair decisions; failures={failures[:3]}")

    semantic_audit: list[dict[str, Any]] = []
    union = UnionFind(len(exact_kept))
    for pair in pairs:
        decision = decisions[pair["pair_id"]]
        semantic_audit.append({**pair, "mechanism_semantic_label": decision["label"], "mechanism_semantic_canonical_phrase": decision["canonical_phrase"], "mechanism_semantic_reason": decision["reason"], "mechanism_semantic_model": args.model})
        if decision["label"] == "duplicate":
            union.union(pair["left_index"], pair["right_index"])
    components: dict[int, list[int]] = {}
    for index in range(len(exact_kept)):
        components.setdefault(union.find(index), []).append(index)
    final_rows: list[dict[str, Any]] = []
    for number, members in enumerate(components.values(), 1):
        members.sort(key=lambda index: (exact_kept[index].get("year", ""), exact_kept[index].get("phrase_decomposition_row_id", "")))
        representative = exact_kept[members[0]]
        canonical = next((semantic_audit_item["mechanism_semantic_canonical_phrase"] for semantic_audit_item in semantic_audit if semantic_audit_item["mechanism_semantic_label"] == "duplicate" and semantic_audit_item["mechanism_semantic_canonical_phrase"] and (semantic_audit_item["left_index"] in members or semantic_audit_item["right_index"] in members)), "")
        final_rows.append({**representative, "mechanism_semantic_group_id": f"MSEM{number:05d}" if len(members) > 1 else "", "mechanism_semantic_group_size": len(members), "mechanism_semantic_canonical_phrase": canonical, "mechanism_semantic_dedup_decision": "kept", "mechanism_semantic_dedup_model": args.model})

    final_rows.sort(key=mechanism_sort_key)
    for number, row in enumerate(final_rows, start=1):
        ticker = normalise(str(row.get("ticker", "firm"))).upper() or "FIRM"
        row["mechanism_phrase_id"] = f"{ticker}-M{number:06d}"

    fields = ["mechanism_phrase_id"] + [field for field in source_fields if field != "mechanism_phrase_id"] + [column for column in EXACT_COLUMNS + SEMANTIC_COLUMNS if column not in source_fields]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mechanism_exact_dedup_audit.csv", source_fields + EXACT_COLUMNS, exact_audit)
    write_csv(args.output_dir / "mechanism_exact_dedup_kept.csv", source_fields + EXACT_COLUMNS, exact_kept)
    write_csv(args.output_dir / "mechanism_semantic_dedup_pair_audit.csv", ["pair_id", "left_phrase", "right_phrase", "left_source_sentence", "right_source_sentence", "left_year", "right_year", "sequence_similarity", "token_jaccard", "mechanism_semantic_label", "mechanism_semantic_canonical_phrase", "mechanism_semantic_reason", "mechanism_semantic_model"], semantic_audit)
    write_csv(args.output_dir / "mechanism_deduped.csv", fields, final_rows)
    manifest = {"input": str(args.input), "input_rows": len(rows), "within_year_only": args.within_year_only, "within_firm_year_only": args.within_firm_year_only, "preserve_material_qualifiers": args.preserve_material_qualifiers, "after_exact": len(exact_kept), "semantic_candidate_pairs": len(pairs), "material_qualifier_pair_exclusions": material_qualifier_pair_exclusions, "semantic_duplicate_pairs": sum(row["mechanism_semantic_label"] == "duplicate" for row in semantic_audit), "conservative_no_merge_pairs": conservative_no_merge_pairs, "after_semantic": len(final_rows), "model": args.model, "failures": failures}
    (args.output_dir / "mechanism_dedup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
