#!/usr/bin/env python3
"""Label residual semantic duplicates and activity actors with DeepSeek.

The script is resumable. Raw batch responses are cached as JSONL, and no API
credential is written to any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cache[item["request_hash"]] = item
    return cache


def append_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    payload: Any,
    cache_path: Path,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    request = {"model": model, "system_prompt": system_prompt, "payload": payload}
    request_hash = digest(request)
    if request_hash in cache and cache[request_hash].get("status") == "ok":
        return cache[request_hash]["response"]
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            record = {
                "status": "ok",
                "request_hash": request_hash,
                "response": parsed,
                "model": model,
            }
            append_cache(cache_path, record)
            cache[request_hash] = record
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def batches(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def duplicate_candidates(frame: pd.DataFrame, embeddings: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    candidates = []
    pair_id = 0
    for _, group in frame.groupby(["firm_id", "year"], sort=False):
        indices = group.index.to_numpy()
        similarity = embeddings[indices] @ embeddings[indices].T
        for left_offset in range(len(indices)):
            for right_offset in range(left_offset + 1, len(indices)):
                score = float(similarity[left_offset, right_offset])
                if score < threshold:
                    continue
                left = int(indices[left_offset])
                right = int(indices[right_offset])
                pair_id += 1
                candidates.append({
                    "pair_id": pair_id,
                    "left_index": left,
                    "right_index": right,
                    "left_row_id": int(frame.at[left, "row_id"]),
                    "right_row_id": int(frame.at[right, "row_id"]),
                    "ticker": str(frame.at[left, "ticker"]),
                    "company": str(frame.at[left, "company"]),
                    "cosine_similarity": score,
                    "left_phrase": str(frame.at[left, "activity_phrase_clean"]),
                    "right_phrase": str(frame.at[right, "activity_phrase_clean"]),
                    "left_source_sentence": str(frame.at[left, "source_sentence"]),
                    "right_source_sentence": str(frame.at[right, "source_sentence"]),
                })
    return candidates


def audit_duplicates(frame, embeddings, client, model, output_dir, cache_dir, threshold, batch_size):
    candidates = duplicate_candidates(frame, embeddings, threshold)
    system_prompt = (
        "You audit possible duplicate firm activities. For each pair, decide whether the two "
        "phrases express the same underlying activity choice for the same focal firm. Preserve "
        "differences in verb/action, object, negation, channel, mode, scope, geography, customer "
        "segment, timing, ownership, in-house versus outsourced status, and product. Use label "
        "duplicate only when one phrase could replace the other without changing the activity's "
        "meaning. Otherwise use related_but_distinct or not_duplicate. Return JSON only as "
        "{\"assessments\":[{\"pair_id\":1,\"label\":\"duplicate|related_but_distinct|not_duplicate\","
        "\"canonical_phrase\":\"...\",\"reason\":\"...\"}]} and include every pair_id."
    )
    cache_path = cache_dir / "deepseek_duplicate_batches.jsonl"
    cache = load_cache(cache_path)
    assessments = []
    for batch in batches(candidates, batch_size):
        payload = [{k: v for k, v in item.items() if k not in {"left_index", "right_index"}} for item in batch]
        parsed = call_json(client, model, system_prompt, payload, cache_path, cache)
        returned = {int(item["pair_id"]): item for item in parsed.get("assessments", [])}
        expected = {item["pair_id"] for item in batch}
        if set(returned) != expected:
            raise RuntimeError(f"DeepSeek duplicate batch IDs mismatch: expected {expected}, got {set(returned)}")
        for item in batch:
            result = returned[item["pair_id"]]
            label = str(result.get("label", "")).strip()
            if label not in {"duplicate", "related_but_distinct", "not_duplicate"}:
                raise RuntimeError(f"Unexpected duplicate label: {label}")
            assessments.append({**item, **{
                "deepseek_duplicate_label": label,
                "deepseek_canonical_phrase": str(result.get("canonical_phrase", "")),
                "deepseek_duplicate_reason": str(result.get("reason", "")),
                "deepseek_duplicate_model": model,
                "deepseek_duplicate_status": "ok",
            }})

    pair_frame = pd.DataFrame(assessments)
    pair_frame.drop(columns=["left_index", "right_index"], errors="ignore").to_csv(
        output_dir / "deepseek_semantic_duplicate_pairs.csv", index=False
    )
    union_find = UnionFind(len(frame))
    for item in assessments:
        if item["deepseek_duplicate_label"] == "duplicate":
            union_find.union(item["left_index"], item["right_index"])
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        components[union_find.find(index)].append(index)
    duplicate_components = [members for members in components.values() if len(members) > 1]
    duplicate_components.sort(key=min)
    row_groups = defaultdict(list)
    for number, members in enumerate(duplicate_components, start=1):
        group_id = f"DSD{number:04d}"
        for index in members:
            row_groups[index].append(group_id)
    labeled = frame.copy()
    labeled["deepseek_has_semantic_duplicate"] = ["yes" if i in row_groups else "no" for i in range(len(frame))]
    labeled["deepseek_semantic_duplicate_group_ids"] = ["|".join(row_groups.get(i, [])) for i in range(len(frame))]
    labeled["deepseek_duplicate_candidate_threshold"] = threshold
    labeled["deepseek_duplicate_model"] = model
    labeled["deepseek_duplicate_status"] = "ok"
    labeled.to_csv(output_dir / "activities_medium_deduplicated_deepseek_labels.csv", index=False)
    return labeled, pair_frame, len(duplicate_components)


def identify_actors(frame, client, model, output_dir, cache_dir, batch_size):
    system_prompt = (
        "Identify who performs each extracted activity using only its source sentence. The focal "
        "company is supplied. actor_class must be exactly one of: firm, external_actor, "
        "product_or_system, unclear. Use firm for the focal company, 'we/the company', or a "
        "clearly named controlled operating unit or subsidiary of that focal company. Use "
        "external_actor for customers, employees acting personally, unions, suppliers, dealers, "
        "competitors, governments, regulators, generic firms, or another company. Use "
        "product_or_system when a product, program, website, network, aircraft, or system is the "
        "actor and firm agency is not explicit. Do not infer ownership or agency from outside "
        "knowledge. Return JSON only as {\"assessments\":[{\"row_id\":1,"
        "\"actor\":\"exact actor text\",\"actor_class\":\"firm|external_actor|product_or_system|unclear\","
        "\"reason\":\"...\"}]} and include every row_id."
    )
    cache_path = cache_dir / "deepseek_actor_batches.jsonl"
    cache = load_cache(cache_path)
    records = frame.to_dict("records")
    actor_results = []
    for batch in batches(records, batch_size):
        payload = [{
            "row_id": int(item["row_id"]),
            "ticker": str(item["ticker"]),
            "focal_company": str(item["company"]),
            "activity_phrase": str(item["activity_phrase_raw"]),
            "source_sentence": str(item["source_sentence"]),
        } for item in batch]
        parsed = call_json(client, model, system_prompt, payload, cache_path, cache)
        returned = {int(item["row_id"]): item for item in parsed.get("assessments", [])}
        expected = {item["row_id"] for item in payload}
        if set(returned) != expected:
            raise RuntimeError(f"DeepSeek actor batch IDs mismatch: expected {expected}, got {set(returned)}")
        for item in payload:
            result = returned[item["row_id"]]
            actor_class = str(result.get("actor_class", "")).strip()
            if actor_class not in {"firm", "external_actor", "product_or_system", "unclear"}:
                raise RuntimeError(f"Unexpected actor class: {actor_class}")
            actor_results.append({
                "row_id": item["row_id"],
                "actor": str(result.get("actor", "")),
                "actor_class": actor_class,
                "actor_reason": str(result.get("reason", "")),
                "actor_model": model,
                "actor_status": "ok",
            })
    actor_frame = pd.DataFrame(actor_results)
    final = frame.merge(actor_frame, on="row_id", how="left", validate="one_to_one")
    final.to_csv(output_dir / "activities_medium_with_actor.csv", index=False)
    final[final["actor_class"].eq("firm")].to_csv(
        output_dir / "activities_medium_actor_firm_only.csv", index=False
    )
    actor_frame.to_csv(output_dir / "deepseek_actor_audit.csv", index=False)
    return final


def parse_args():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project / "phrases/clean_normalize/deduplication/activities_medium_deduplicated.csv")
    parser.add_argument("--embeddings", type=Path, default=project / "phrases/clean_normalize/deduplication/deduplicated_activity_embeddings.npy")
    parser.add_argument("--dedup-output-dir", type=Path, default=project / "phrases/clean_normalize/deduplication")
    parser.add_argument("--actor-output-dir", type=Path, default=project / "phrases/clean_normalize/actor_identification")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--duplicate-batch-size", type=int, default=12)
    parser.add_argument("--actor-batch-size", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    frame["row_id"] = frame["row_id"].astype(int)
    frame = frame.reset_index(drop=True)
    embeddings = np.load(args.embeddings)
    if embeddings.shape[0] != len(frame):
        raise RuntimeError("Embedding rows do not match deduplicated activity rows")
    args.dedup_output_dir.mkdir(parents=True, exist_ok=True)
    args.actor_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.dedup_output_dir / "api_audit"
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120)
    labeled, pairs, n_groups = audit_duplicates(
        frame, embeddings, client, args.model, args.dedup_output_dir, cache_dir,
        args.duplicate_threshold, args.duplicate_batch_size,
    )
    final = identify_actors(
        labeled, client, args.model, args.actor_output_dir, cache_dir, args.actor_batch_size
    )
    summary = pd.DataFrame([
        {"metric": "deduplicated_input_rows", "value": len(frame)},
        {"metric": "deepseek_candidate_pairs", "value": len(pairs)},
        {"metric": "deepseek_duplicate_pairs", "value": int((pairs["deepseek_duplicate_label"] == "duplicate").sum())},
        {"metric": "deepseek_duplicate_groups", "value": n_groups},
        {"metric": "actor_rows", "value": len(final)},
        {"metric": "firm_actor_rows", "value": int(final["actor_class"].eq("firm").sum())},
        {"metric": "model", "value": args.model},
    ])
    summary.to_csv(args.actor_output_dir / "deepseek_labeling_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
