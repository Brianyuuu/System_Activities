#!/usr/bin/env python3
"""Atomize already-produced core mechanism phrases and expand one row per atom.

Only ``core_mechanism_phrase`` is sent to the model.  Firm, ticker, year,
source sentence, actor fields, and taxonomy labels remain outside the prompt.
They are copied back only after the model result has been validated.
"""
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
from typing import Any

from openai import OpenAI


BARE_FRAGMENT_RE = re.compile(
    r"^(?:and|or|but|from|under|on(?!-)|through|via|with|at|in|by|for|to)\b",
    re.IGNORECASE,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).rstrip(".")


def is_placeholder_rationale(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return normalized in {"", "real words", "real rationale", "rationale here", "placeholder"}


def singleton_json_regex(item_id: int, parent_phrase: str = "") -> str:
    """Same singleton schema, with no grammar path for inter-field padding."""
    json_character = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
    json_string = r'"' + json_character + r'*"'
    # Recovery reasons must mention the supplied activity, not generic placeholders.
    stop_words = {"this", "that", "with", "from", "into", "their", "other", "only",
                  "there", "have", "been", "which", "will", "would", "could", "should"}
    words = sorted(set(re.findall(r"[a-z]{4,}", parent_phrase.casefold())) - stop_words)
    grounded_word = ("(?:" + "|".join("".join(f"[{c}{c.upper()}]" for c in word)
                                     for word in words) + ")") if words else r"[A-Za-z]"
    rationale_string = r'"' + json_character + "*" + grounded_word + json_character + r'*"'
    return (r'\{"items":\[\{"id":' + str(item_id) + r',"atom_mechanism_phrases":\['
            + json_string + r'(?:,' + json_string + r'){0,11}\],"decision":"(?:unchanged|split)",'
            + r'"confidence":"(?:high|medium|low)","rationale":' + rationale_string + r'\}\]\}')


def is_bare_fragment(value: str) -> bool:
    # These hyphenated verbs are not "in ..." prepositional fragments.
    if re.match(r"^in[-\u2010\u2011](?:(?:licens|sourc)(?:e|ing)|fill(?:ing)?)\b", value, re.IGNORECASE):
        return False
    if not BARE_FRAGMENT_RE.match(value):
        return False
    # A fronted scope/condition followed by a comma and an action predicate is
    # a complete mechanism (for example, "for Solar Roof, provide warranties").
    _, separator, predicate = value.partition(",")
    if separator and len(predicate.strip().split()) >= 2:
        return False
    return True


def validate(
    payload: Any,
    expected_ids: set[int],
    parent_phrases: dict[int, str],
) -> dict[int, dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Response lacks items list")
    result: dict[int, dict[str, Any]] = {}
    required = {
        "id",
        "atom_mechanism_phrases",
        "decision",
        "confidence",
        "rationale",
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("Each item has an invalid schema")
        item_id = item["id"]
        if (
            len(expected_ids) == 1 and len(items) == 1
            and type(item_id) is int and item_id in {0, 1}
            and item_id not in expected_ids
        ):
            # GLM sometimes copies the schema illustration's id despite
            # generating useful atoms for the only supplied phrase. With one
            # input row, its identity is unambiguous; content checks below
            # still reject copied placeholder atoms and rationale.
            actual_id = next(iter(expected_ids))
            print(f"Normalized illustrative singleton id {item_id} to {actual_id}", flush=True)
            item_id = actual_id
        if type(item_id) is not int or item_id not in expected_ids or item_id in result:
            raise ValueError(f"Unexpected or duplicate id: {item_id!r}")
        raw_atoms = item["atom_mechanism_phrases"]
        if not isinstance(raw_atoms, list) or not raw_atoms:
            raise ValueError(f"Item {item_id} has no atom mechanism phrases")
        atoms: list[str] = []
        seen: set[str] = set()
        for raw_atom in raw_atoms:
            if not isinstance(raw_atom, str):
                raise ValueError(f"Item {item_id} has a non-string atom")
            atom = clean_phrase(raw_atom)
            if not atom:
                raise ValueError(f"Item {item_id} has an empty atom")
            if atom in {"..", "...", "…"}:
                raise ValueError(f"Item {item_id} returned a placeholder atom")
            if is_bare_fragment(atom):
                raise ValueError(f"Item {item_id} returned an incomplete fragment: {atom!r}")
            key = atom.casefold()
            if key not in seen:
                seen.add(key)
                atoms.append(atom)
        declared_decision = item["decision"]
        if declared_decision not in {"unchanged", "split"}:
            raise ValueError(f"Item {item_id} has invalid decision")
        # Atom count is authoritative.  Some otherwise valid model responses
        # label a multi-atom list "unchanged"; normalizing that metadata avoids
        # discarding grounded atoms or retrying an identical semantic result.
        decision = "split" if len(atoms) > 1 else "unchanged"
        confidence = item["confidence"]
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Item {item_id} has invalid confidence")
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Item {item_id} has empty rationale")
        if is_placeholder_rationale(rationale):
            raise ValueError(f"Item {item_id} has placeholder rationale")
        if len(atoms) > 1 and any(
            atom.casefold() == parent_phrases[item_id].casefold() for atom in atoms
        ):
            raise ValueError(f"Item {item_id} retained the full composite parent as an atom")
        result[item_id] = {
            "atom_mechanism_phrases": atoms,
            "decision": decision,
            "confidence": confidence,
            "rationale": clean_phrase(rationale),
        }
    if set(result) != expected_ids:
        raise ValueError(f"Response coverage mismatch: expected {sorted(expected_ids)}")
    return result


def reuse_prior_atoms(
    source_fields: list[str], indexed: list[tuple[int, dict[str, str]]],
    previous_rows: list[dict[str, str]], model: str,
) -> dict[int, dict[str, Any]]:
    """Reuse complete successful parent groups only when copied input fields match."""
    ids = [row.get("phrase_decomposition_row_id", "") for _, row in indexed]
    if not all(ids) or len(set(ids)) != len(ids):
        return {}
    groups: dict[str, list[dict[str, str]]] = {}
    for previous in previous_rows:
        groups.setdefault(previous.get("phrase_decomposition_row_id", ""), []).append(previous)
    overwritten = {"core_mechanism_phrase", "activity_phrase", "replaced_activity_phrase"}
    reusable: dict[int, dict[str, Any]] = {}
    for index, current in indexed:
        group = groups.get(current["phrase_decomposition_row_id"], [])
        parent = clean_phrase(current["core_mechanism_phrase"])
        if not group or any(
            previous.get("parent_core_mechanism_phrase") != parent
            or previous.get("core_atomization_model") != model
            or "Atomization API failed" in previous.get("core_atomization_rationale", "")
            or previous.get("pre_split_replaced_activity_phrase", "") != current.get("replaced_activity_phrase", "")
            or any(previous.get(field, "") != current.get(field, "")
                   for field in source_fields if field not in overwritten)
            for previous in group
        ):
            continue
        try:
            ordered = sorted(group, key=lambda row: int(row["atom_phrase_index"]))
            if [int(row["atom_phrase_index"]) for row in ordered] != list(range(1, len(ordered)+1)):
                continue
            if any(int(row["atom_phrase_count"]) != len(ordered) for row in ordered):
                continue
            first = ordered[0]
            if any(row.get(field) != first.get(field) for row in ordered
                   for field in ("core_atomization_decision", "core_atomization_confidence", "core_atomization_rationale")):
                continue
            atoms = [row["atom_mechanism_phrase"] for row in ordered]
            if any(json.loads(row["atom_mechanism_phrases_json"]) != atoms for row in ordered):
                continue
            checked = validate({"items": [{
                "id": index, "atom_mechanism_phrases": atoms,
                "decision": first["core_atomization_decision"],
                "confidence": first["core_atomization_confidence"],
                "rationale": first["core_atomization_rationale"],
            }]}, {index}, {index: parent})
            reusable.update(checked)
        except (ValueError, KeyError, TypeError):
            continue
    return reusable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reuse-successful-output", action="store_true",
                        default=os.getenv("PHRASE_REPAIR_REUSE_OUTPUT", "") == "1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--bisect-on-failure",
        action="store_true",
        help="Split a failed multi-row request into smaller requests before retrying.",
    )
    parser.add_argument(
        "--fallback-to-parent-on-failure",
        action="store_true",
        help="Preserve a failed singleton parent for the downstream atomic audit.",
    )
    args = parser.parse_args()

    if (
        args.workers < 1
        or args.batch_size < 1
        or args.max_tokens < 1
        or args.request_timeout <= 0
    ):
        raise SystemExit("workers, batch-size, max-tokens, and request-timeout must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields, rows = list(reader.fieldnames or []), list(reader)
    if "core_mechanism_phrase" not in source_fields:
        raise SystemExit("Input must contain core_mechanism_phrase")
    if any(not clean_phrase(row.get("core_mechanism_phrase", "")) for row in rows):
        raise SystemExit("Input contains an empty core_mechanism_phrase")

    prompt = args.prompt.read_text(encoding="utf-8")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=args.request_timeout,
    )
    indexed = list(enumerate(rows))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    previous_fallback_parents: set[str] = set()
    previous_rows: list[dict[str, str]] = []
    previous_output_sha256 = None
    if args.output.is_file():
        previous_output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
        with args.output.open("r", encoding="utf-8", newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
            for previous_row in previous_rows:
                if ("Atomization API failed" in previous_row.get("core_atomization_rationale", "")
                        or is_placeholder_rationale(previous_row.get("core_atomization_rationale", ""))):
                    parent = clean_phrase(previous_row.get("parent_core_mechanism_phrase", ""))
                    if parent:
                        previous_fallback_parents.add(parent)
    reused_results = reuse_prior_atoms(source_fields, indexed, previous_rows, args.model) if args.reuse_successful_output else {}
    remaining = [(index, row) for index, row in indexed if index not in reused_results]
    batches = [remaining[start : start + args.batch_size] for start in range(0, len(remaining), args.batch_size)]
    if args.reuse_successful_output:
        print(f"Reusing validated atomization output for {len(reused_results)}/{len(indexed)} parent phrases; requesting {len(remaining)}", flush=True)

    def work(
        batch: list[tuple[int, dict[str, str]]],
        retry_limit: int,
    ) -> dict[int, dict[str, Any]]:
        # Intentionally exclude firm, ticker, year, source context, and all other
        # record metadata from the model-visible payload.
        model_items = [
            {"id": index, "core_mechanism_phrase": clean_phrase(row["core_mechanism_phrase"])}
            for index, row in batch
        ]
        glm_singleton_note = ""
        effective_prompt = prompt
        if args.model == "parcc/GLM" and len(model_items) == 1:
            example = ('{"items":[{"id":0,"atom_mechanism_phrases":["..."],'
                       '"decision":"unchanged|split","confidence":"high|medium|low",'
                       '"rationale":"..."}]}')
            if example not in prompt:
                raise ValueError("GLM atomization example to remove was not found")
            effective_prompt = prompt.replace(
                example, "Return an items array matching the response schema."
            )
            item_id = model_items[0]["id"]
            glm_singleton_note = (
                "\n\nEcho exactly the supplied input id "
                f"{item_id}, and write grounded atom phrases and a real "
                "rationale. Never output literal ellipses."
            )
        compact_glm_prompt = (
            "Split the input activity phrase into complete, distinct activity phrases only if it "
            "contains separately asserted actions or channels. Each atom needs its own verb and "
            "object. Preserve material qualifiers. Do not add facts. Return one item with exactly "
            f"the supplied id {model_items[0]['id']}; use the JSON schema. "
            "Write a brief factual rationale explaining why this input is one activity or was split. "
            "The rationale must contain real words from this input, never punctuation-only text "
            "or placeholder dots. Do not output ellipses."
        ) if args.model == "parcc/GLM" and len(model_items) == 1 else ""
        legacy_request_hash = hashlib.sha256(
            json.dumps(
                {"model": args.model, "prompt": effective_prompt, "items": model_items,
                 "prompt_format": (
                     "single_user_schema_v3_singleton_no_example" if glm_singleton_note else
                     "single_user_schema_v1" if args.model == "parcc/GLM" else "system_user"
                 )},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        json_layout_note = (
            "\nEmit compact JSON with no whitespace outside JSON string values. "
            "Never pad between a colon and its value."
        ) if args.model == "parcc/GLM" else ""
        primary_prompt = effective_prompt + glm_singleton_note + json_layout_note
        if compact_glm_prompt:
            compact_glm_prompt += json_layout_note
        request_context = {
            "legacy_request_hash": legacy_request_hash,
            "primary_prompt": primary_prompt,
            "recovery_prompt": compact_glm_prompt,
            "prompt_format": "compact_json_layout_v1",
            "recovery_constraint": "singleton_compact_regex_v3_grounded_rationale",
        }
        request_hash = hashlib.sha256(json.dumps(request_context, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() if json_layout_note else legacy_request_hash
        previous_request_hashes = [hashlib.sha256(json.dumps(
            {**request_context, "recovery_constraint": version},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            for version in ("singleton_compact_regex_v2_rationale_words", "singleton_compact_regex_v1")
        ] if json_layout_note else []
        cache_file = args.cache_dir / f"{request_hash}.json"
        expected = {index for index, _ in batch}
        parents = {index: clean_phrase(row["core_mechanism_phrase"]) for index, row in batch}
        # Formatting-only changes do not invalidate already validated activities.
        # A stale/invalid cache must never bypass the API recovery attempts.
        for candidate_hash in dict.fromkeys((request_hash, *previous_request_hashes, legacy_request_hash)):
            candidate_file = args.cache_dir / f"{candidate_hash}.json"
            if candidate_file.exists():
                try:
                    cached = json.loads(candidate_file.read_text(encoding="utf-8"))
                    if cached.get("request_hash") == candidate_hash:
                        return validate(cached.get("response"), expected, parents)
                except (ValueError, OSError) as cache_error:
                    print(f"Ignoring invalid atomization cache {candidate_hash[:12]}: {type(cache_error).__name__}", flush=True)

        last_error: Exception | None = None
        token_limit = args.max_tokens
        use_compact_glm_prompt = bool(
            compact_glm_prompt and parents[next(iter(expected))] in previous_fallback_parents
        )
        force_regex_glm = use_compact_glm_prompt
        if force_regex_glm:
            token_limit = min(args.max_tokens, 1024)
        for attempt in range(retry_limit + 1):
            response = None
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    temperature=0.0,
                    max_tokens=token_limit,
                    **({"response_format": (
                        {"type": "json_schema", "json_schema": {
                            "name": "atomic_items", "strict": True,
                            "schema": {"type": "object", "additionalProperties": False,
                                       "properties": {"items": {"type": "array",
                                           "minItems": len(model_items), "maxItems": len(model_items),
                                           "items": {
                                           "type": "object", "additionalProperties": False,
                                           "properties": {
                                               "id": {"type": "integer"},
                                               "atom_mechanism_phrases": {"type": "array", "items": {"type": "string"},
                                                                          "minItems": 1, "maxItems": 12},
                                               "decision": {"type": "string", "enum": ["unchanged", "split"]},
                                               "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                               "rationale": {"type": "string"},
                                           },
                                           "required": ["id", "atom_mechanism_phrases", "decision", "confidence", "rationale"],
                                       }}}, "required": ["items"]},
                        }} if args.model == "parcc/GLM" else {"type": "json_object"}
                    )} if not force_regex_glm else {}),
                    messages=(
                        [{"role": "user", "content": (
                            compact_glm_prompt if use_compact_glm_prompt else primary_prompt
                        ) + "\n\nINPUT ITEMS:\n" + json.dumps({"items": model_items}, ensure_ascii=False)}]
                        if args.model == "parcc/GLM" else [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({"items": model_items}, ensure_ascii=False)},
                        ]
                    ),
                    extra_body={
                        "thinking": {"type": "disabled"},
                        **({"chat_template_kwargs": {"enable_thinking": False}} if args.model == "parcc/GLM" else {}),
                        **({"structured_outputs": {"regex": singleton_json_regex(next(iter(expected)), parents[next(iter(expected))])}}
                           if force_regex_glm else {}),
                    },
                )
                if response.choices[0].finish_reason == "length":
                    raise ValueError("Atomization response truncated at token limit")
                message = response.choices[0].message
                parsed = json.loads((message.content or "").strip() or (message.model_extra or {}).get("reasoning_content") or "{}")
                checked = validate(parsed, expected, parents)
                atomic_json(cache_file, {
                    "request_hash": request_hash,
                    "prompt_variant": "compact_recovery" if use_compact_glm_prompt else "primary",
                    "constraint_mode": "singleton_compact_regex" if force_regex_glm else "json_schema",
                    "response": parsed,
                })
                return checked
            except Exception as error:
                last_error = error
                longest_whitespace = 0
                if response is not None:
                    choice = response.choices[0]
                    message = choice.message
                    content = message.content or ""
                    reasoning = (message.model_extra or {}).get("reasoning_content") or ""
                    selected = content if content.strip() else reasoning
                    longest_whitespace = max((len(s) for s in re.findall(r"[ \t\r\n]+", selected)), default=0)
                    usage = getattr(response, "usage", None)
                    diagnostic = {
                        "request_hash": request_hash, "model": args.model,
                        "input_items": model_items, "attempt": attempt + 1,
                        "prompt_variant": "compact_recovery" if use_compact_glm_prompt else "primary",
                        "constraint_mode": "singleton_compact_regex" if force_regex_glm else "json_schema",
                        "max_tokens": token_limit, "finish_reason": choice.finish_reason,
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "error_type": type(error).__name__, "error": str(error)[:400],
                        "json_error_pos": getattr(error, "pos", None),
                        "longest_whitespace_run": longest_whitespace,
                        "content": content, "reasoning_content": reasoning,
                    }
                    try:
                        atomic_json(args.cache_dir / "invalid_responses" / f"{request_hash}_{time.time_ns()}.json", diagnostic)
                    except OSError:
                        print("Could not persist atomization response diagnostic", flush=True)
                malformed_or_truncated = (
                    isinstance(error, ValueError)
                    or "truncated at token limit" in str(error)
                )
                if malformed_or_truncated and compact_glm_prompt and longest_whitespace >= 64:
                    use_compact_glm_prompt = True
                    force_regex_glm = True
                    token_limit = min(args.max_tokens, 1024)
                    print("Retrying singleton atomization after excessive JSON whitespace with compact regex and bounded tokens", flush=True)
                elif malformed_or_truncated and compact_glm_prompt and not use_compact_glm_prompt:
                    use_compact_glm_prompt = True
                    print("Retrying singleton atomization with concise GLM prompt", flush=True)
                elif malformed_or_truncated:
                    token_limit = min(16000, max(token_limit * 2, args.max_tokens))
                if attempt < retry_limit:
                    time.sleep(min(15.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Core phrase atomization failed: {last_error}")

    def work_with_recovery(
        batch: list[tuple[int, dict[str, str]]],
    ) -> dict[int, dict[str, Any]]:
        # A malformed or stalled response for one row must not force the same
        # multi-row payload to be retried for tens of minutes.  Cached requests
        # still return immediately.  A new multi-row request gets one attempt;
        # on failure it is bisected until the problematic row is isolated.
        retry_limit = 0 if args.bisect_on_failure and len(batch) > 1 else args.max_retries
        try:
            return work(batch, retry_limit)
        except Exception as error:
            if len(batch) == 1 and args.fallback_to_parent_on_failure:
                index, row = batch[0]
                parent = clean_phrase(row["core_mechanism_phrase"])
                print(
                    f"Core atomization row {index + 1} failed after validated retries "
                    f"({type(error).__name__}: {str(error)[:240]}); "
                    "preserving parent for Stage 7 audit",
                    flush=True,
                )
                return {
                    index: {
                        "atom_mechanism_phrases": [parent],
                        "decision": "unchanged",
                        "confidence": "low",
                        "rationale": (
                            "Atomization API failed after validated retries; parent phrase "
                            "preserved for downstream contextual atomic audit"
                        ),
                    }
                }
            if not args.bisect_on_failure or len(batch) == 1:
                raise
            midpoint = len(batch) // 2
            left, right = batch[:midpoint], batch[midpoint:]
            first_id, last_id = batch[0][0], batch[-1][0]
            print(
                f"Core atomization request for rows {first_id + 1}-{last_id + 1} "
                f"failed ({type(error).__name__}); retrying as "
                f"{len(left)}+{len(right)} rows",
                flush=True,
            )
            recovered: dict[int, dict[str, Any]] = {}
            for smaller_batch in (left, right):
                smaller_result = work_with_recovery(smaller_batch)
                overlap = set(recovered) & set(smaller_result)
                if overlap:
                    raise RuntimeError(f"Duplicate recovered atomization results: {sorted(overlap)}")
                recovered.update(smaller_result)
            return recovered

    results: dict[int, dict[str, Any]] = dict(reused_results)
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(batches)))) as executor:
        futures = [executor.submit(work_with_recovery, batch) for batch in batches]
        for number, future in enumerate(as_completed(futures), 1):
            batch_result = future.result()
            overlap = set(results) & set(batch_result)
            if overlap:
                raise RuntimeError(f"Duplicate atomization results: {sorted(overlap)}")
            results.update(batch_result)
            print(f"Core atomization batches: {number}/{len(batches)}", flush=True)
    if len(results) != len(rows):
        raise RuntimeError("Core atomization result count mismatch")

    added_fields = [
        "parent_core_mechanism_phrase",
        "atom_mechanism_phrase",
        "atom_mechanism_phrases_json",
        "atom_phrase_index",
        "atom_phrase_count",
        "core_atomization_decision",
        "core_atomization_confidence",
        "core_atomization_rationale",
        "core_atomization_model",
        "pre_split_replaced_activity_phrase",
    ]
    output_rows: list[dict[str, Any]] = []
    for index, row in indexed:
        parent = clean_phrase(row["core_mechanism_phrase"])
        atomization = results[index]
        atoms = atomization["atom_mechanism_phrases"]
        for position, atom in enumerate(atoms, 1):
            output_rows.append(
                {
                    **row,
                    "parent_core_mechanism_phrase": parent,
                    "atom_mechanism_phrase": atom,
                    "atom_mechanism_phrases_json": json.dumps(atoms, ensure_ascii=False),
                    "atom_phrase_index": position,
                    "atom_phrase_count": len(atoms),
                    "core_atomization_decision": atomization["decision"],
                    "core_atomization_confidence": atomization["confidence"],
                    "core_atomization_rationale": atomization["rationale"],
                    "core_atomization_model": args.model,
                    "pre_split_replaced_activity_phrase": row.get("replaced_activity_phrase", ""),
                    # Downstream atomic audit consumes core_mechanism_phrase.
                    "core_mechanism_phrase": atom,
                    "activity_phrase": atom,
                    "replaced_activity_phrase": atom,
                }
            )
    fields = source_fields + [field for field in added_fields if field not in source_fields]
    write_csv(args.output, fields, output_rows)
    manifest = {
        "input": str(args.input),
        "parent_core_phrases": len(rows),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "reused_parent_phrases": len(reused_results),
        "prior_output_sha256": previous_output_sha256 if reused_results else None,
        "atom_mechanism_phrases": len(output_rows),
        "split_parent_phrases": sum(result["decision"] == "split" for result in results.values()),
        "model_visible_fields": ["id", "core_mechanism_phrase"],
        "firm_year_examples_used": False,
        "model": args.model,
    }
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
