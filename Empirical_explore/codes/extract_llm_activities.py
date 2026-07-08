#!/usr/bin/env python3
"""Extract explicitly stated Item 1 activities with DeepSeek JSON mode.

The API key is read only from DEEPSEEK_API_KEY. Raw responses and validation
decisions are retained so the extraction is auditable and resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spacy
from openai import OpenAI

from extract_dependency_phrases import extract_item1, filing_from_path, html_to_text


BASE_PROMPT = """Read the following 10-K Item 1 business description. Extract every distinct operational activity the firm explicitly describes itself as performing. Be specific: ‘manufactures cars’ is better than ‘manufactures.’ Do not infer activities not described. Do not use outside knowledge. Only extract what is stated. For each activity, return the exact source sentence that supports it. If no explicit operational activity appears, return an empty list."""

JSON_INSTRUCTION = """Return one JSON object with exactly this structure:
{"activities": [{"activity_phrase": "...", "source_sentence": "..."}]}
`source_sentence` must be copied character-for-character from the input chunk and must be one complete sentence. Do not add fields or commentary.

INPUT CHUNK:
"""

SENSITIVITY_INSTRUCTIONS = {
    "strict": "",
    "medium": """MEDIUM SENSITIVITY: Increase recall while remaining grounded in the text. Include operational activities stated through coordinated verbs or objects, passive voice, prepositional constructions, and longer noun phrases. Preserve strategically meaningful modifiers such as in-house versus outsourced, direct versus intermediated, channel, mode, scope, exclusivity, timing, and customer segment. Include value creation, delivery, coordination, and value-capture activities. Exclude resources, capabilities, outcomes, risks, and aspirations that do not state an activity the firm performs.""",
    "loose": """LOOSE SENSITIVITY: Use the medium rules and broaden recall further. Include borderline but textually explicit operating practices, organizational and coordination choices, selling and pricing practices, investment or expansion actions, and explicitly stated planned or continuing activities. Retain an item only when the source sentence itself states the action; never infer from firm identity, industry knowledge, resources, outcomes, or common sense.""",
}
SENSITIVITIES = ("strict", "medium", "loose")

FIELDNAMES = [
    "firm_id",
    "company",
    "year",
    "accession",
    "source_method",
    "extraction_sensitivity",
    "model",
    "chunk_id",
    "activity_phrase",
    "source_sentence",
    "source_file",
]


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_chunks(sentences: list[str], max_chars: int) -> list[list[str]]:
    """Create non-overlapping, sentence-bounded chunks."""
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
        current_chars += len(sentence) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["request_hash"]] = record
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def call_llm(
    client: OpenAI,
    model: str,
    chunk: str,
    sensitivity_instruction: str,
    max_retries: int,
) -> tuple[dict[str, Any], str | None, dict[str, int | None]]:
    prompt = BASE_PROMPT
    if sensitivity_instruction:
        prompt += "\n\n" + sensitivity_instruction
    user_content = prompt + "\n\n" + JSON_INSTRUCTION + chunk
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Extract solely from the supplied text.",
                    },
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=16_000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            usage = response.usage
            return parsed, content, {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception as exc:  # API, transport, or malformed JSON
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(min(30, (2**attempt) + random.random()))
    assert last_error is not None
    raise last_error


def validate_response(
    payload: dict[str, Any], exact_sentences: set[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    activities = payload.get("activities", []) if isinstance(payload, dict) else []
    if not isinstance(activities, list):
        return [], [{"reason": "activities_not_a_list", "value": repr(activities)}]
    for activity in activities:
        if not isinstance(activity, dict):
            rejected.append({"reason": "activity_not_an_object", "value": repr(activity)})
            continue
        phrase = activity.get("activity_phrase")
        sentence = activity.get("source_sentence")
        if not isinstance(phrase, str) or not phrase.strip():
            rejected.append({"reason": "missing_activity_phrase", "value": repr(activity)})
        elif not isinstance(sentence, str) or sentence not in exact_sentences:
            rejected.append(
                {
                    "reason": "source_sentence_not_exact",
                    "activity_phrase": phrase,
                    "source_sentence": sentence if isinstance(sentence, str) else repr(sentence),
                }
            )
        else:
            accepted.append(
                {"activity_phrase": phrase.strip(), "source_sentence": sentence}
            )
    return accepted, rejected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["activity_phrase"].casefold(), row["source_sentence"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def nested_union(
    inherited: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    sensitivity: str,
) -> list[dict[str, Any]]:
    relabeled = [dict(row, extraction_sensitivity=sensitivity) for row in inherited]
    relabeled.extend(dict(row, extraction_sensitivity=sensitivity) for row in new_rows)
    return dedupe_rows(relabeled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
    )
    parser.add_argument("--chunk-chars", type=int, default=12_000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--sensitivity",
        choices=["all", *SENSITIVITIES],
        default="all",
        help="Extraction sensitivity; all creates nested strict, medium, and loose outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    paths = sorted(args.input_dir.glob("*_10-k_*.htm"))
    if not paths:
        raise SystemExit(f"No filing HTML files found in {args.input_dir}")

    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120.0)
    cache_path = args.audit_dir / "deepseek_chunk_responses.jsonl"
    cache = load_cache(cache_path)
    modes = SENSITIVITIES if args.sensitivity == "all" else (args.sensitivity,)
    pooled: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    company_summaries: list[dict[str, Any]] = []

    for path in paths:
        filing = filing_from_path(path)
        item1 = extract_item1(html_to_text(path))
        sentences = [sentence.text.strip() for sentence in nlp(item1).sents if sentence.text.strip()]
        chunks = make_chunks(sentences, args.chunk_chars)
        raw_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
        rejected_by_mode: dict[str, int] = {mode: 0 for mode in modes}

        for mode in modes:
            sensitivity_instruction = SENSITIVITY_INSTRUCTIONS[mode]
            for chunk_number, chunk_sentences in enumerate(chunks, start=1):
                chunk_id = f"{path.stem}_chunk_{chunk_number:03d}"
                chunk = "\n".join(chunk_sentences)
                request_spec = {
                    "model": args.model,
                    "base_prompt": BASE_PROMPT,
                    "json_instruction": JSON_INSTRUCTION,
                    "chunk": chunk,
                }
                # Keep strict hashes backward compatible with the original run.
                if sensitivity_instruction:
                    request_spec["sensitivity_instruction"] = sensitivity_instruction
                request_hash = sha256(
                    json.dumps(request_spec, ensure_ascii=False, sort_keys=True)
                )

                if request_hash in cache and cache[request_hash].get("status") == "ok":
                    record = cache[request_hash]
                    payload = record["parsed_response"]
                else:
                    payload, raw_content, usage = call_llm(
                        client,
                        args.model,
                        chunk,
                        sensitivity_instruction,
                        args.max_retries,
                    )
                    accepted, rejected = validate_response(payload, set(chunk_sentences))
                    record = {
                        "status": "ok",
                        "request_hash": request_hash,
                        "chunk_id": chunk_id,
                        "company": filing.company,
                        "model": args.model,
                        "extraction_sensitivity": mode,
                        "chunk_sha256": sha256(chunk),
                        "sentence_count": len(chunk_sentences),
                        "parsed_response": payload,
                        "raw_response": raw_content,
                        "accepted": accepted,
                        "rejected": rejected,
                        "usage": usage,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(cache_path, record)
                    cache[request_hash] = record

                accepted, rejected = validate_response(payload, set(chunk_sentences))
                rejected_by_mode[mode] += len(rejected)
                for activity in accepted:
                    raw_by_mode[mode].append(
                        {
                            "firm_id": filing.firm_id,
                            "company": filing.company,
                            "year": filing.year,
                            "accession": filing.accession,
                            "source_method": "deepseek_llm_chunk_extraction",
                            "extraction_sensitivity": mode,
                            "model": args.model,
                            "chunk_id": chunk_id,
                            "activity_phrase": activity["activity_phrase"],
                            "source_sentence": activity["source_sentence"],
                            "source_file": filing.path.name,
                        }
                    )

        final_by_mode: dict[str, list[dict[str, Any]]] = {}
        for index, mode in enumerate(modes):
            current = dedupe_rows(raw_by_mode[mode])
            if args.sensitivity == "all" and index > 0:
                current = nested_union(final_by_mode[modes[index - 1]], current, mode)
            final_by_mode[mode] = current

        slug = path.name.split("_", 1)[0].lower()
        for mode in modes:
            rows = final_by_mode[mode]
            write_csv(args.output_dir / f"{slug}_deepseek_{mode}_2015.csv", rows)
            write_json(
                args.output_dir / f"{slug}_deepseek_{mode}_2015.json",
                {
                    "company": filing.company,
                    "firm_id": filing.firm_id,
                    "year": filing.year,
                    "accession": filing.accession,
                    "model": args.model,
                    "extraction_sensitivity": mode,
                    "activities": [
                        {
                            "activity_phrase": row["activity_phrase"],
                            "source_sentence": row["source_sentence"],
                        }
                        for row in rows
                    ],
                },
            )
            pooled[mode].extend(rows)
        company_summaries.append(
            {
                "company": filing.company,
                "item1_words": len(item1.split()),
                "sentence_count": len(sentences),
                "chunk_count": len(chunks),
                "accepted_activities": {
                    mode: len(final_by_mode[mode]) for mode in modes
                },
                "rejected_activities": rejected_by_mode,
            }
        )
        counts = ", ".join(f"{mode}={len(final_by_mode[mode])}" for mode in modes)
        print(f"{filing.company}: chunks={len(chunks)}, {counts}")

    combined: list[dict[str, Any]] = []
    for mode in modes:
        write_csv(
            args.output_dir / f"all_companies_deepseek_{mode}_2015.csv",
            pooled[mode],
        )
        combined.extend(pooled[mode])
    write_csv(args.output_dir / "all_companies_deepseek_combined_long_2015.csv", combined)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "chunk_chars": args.chunk_chars,
        "prompt": BASE_PROMPT,
        "json_instruction": JSON_INSTRUCTION,
        "sensitivity_instructions": {
            mode: SENSITIVITY_INSTRUCTIONS[mode] for mode in modes
        },
        "companies": company_summaries,
        "total_accepted_activities": {mode: len(pooled[mode]) for mode in modes},
        "credential_source": "DEEPSEEK_API_KEY environment variable",
    }
    write_json(args.output_dir / "deepseek_extraction_manifest.json", manifest)
    print(f"Combined long format: {len(combined)} validated activity rows")


if __name__ == "__main__":
    main()
