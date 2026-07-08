#!/usr/bin/env python3
"""Clean, classify, embed, and near-deduplicate medium activity candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.85

PRESERVE_WORDS = {
    "direct", "franchised", "electric", "low-cost", "premium",
    "point-to-point", "hub-and-spoke", "digital", "owned", "exclusive",
    "same-day", "store-based", "regional", "international", "through",
    "with", "by", "for", "under", "via", "using", "not", "only",
    "online", "in-house", "outsourced", "domestic", "global", "independent",
}

PROTECTED_MODIFIERS = {
    "direct", "franchised", "electric", "low-cost", "premium",
    "point-to-point", "hub-and-spoke", "digital", "owned", "exclusive",
    "same-day", "store-based", "regional", "international", "online",
    "in-house", "outsourced", "domestic", "global", "self-service",
    "company-owned", "independent-agent", "through-dealers", "through-brokers",
    "employee-salesforce",
}

GENERIC_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "we", "our", "us",
    "its", "their", "it", "they", "them", "he", "she", "i", "you", "of",
    "and", "or", "as", "at", "to", "from", "into", "on", "in", "over",
    "about", "approximately", "currently", "also", "generally", "primarily",
    "certain", "various", "such", "other", "more", "most", "many", "each",
}

FIRM_ACTOR = re.compile(
    r"^(?:\d+\s+)?(?:general\s+)?(?:we|the company|southwest|tesla|ford|delta)\b",
    re.I,
)
AFFILIATE_ACTOR = re.compile(r"^(?:ford credit|monroe|our subsidiaries?)\b", re.I)
EXTERNAL_ACTOR = re.compile(
    r"^(?:eligible\s+)?(?:customers?|members?|drivers?|passengers?|dealers?|"
    r"suppliers?|travel agents?|airports?|governments?|regulators?|competitors?|"
    r"holders?|employees?|unions?|labor unions?|third parties?|congress|"
    r"the president|u\.s\. president|nmb|national mediation board|epa|faa|"
    r"individuals?|companies|program participants?|the parties|either party|"
    r"certified labor union|foreign carriers?|other carriers?)\b",
    re.I,
)
PRODUCT_ACTOR = re.compile(
    r"^(?:model [sx3]|vehicles?|aircraft|the system|our system|software|"
    r"the program|skymiles|rapid rewards|battery packs?)\b",
    re.I,
)

FIRM_UNIT_ACTOR = re.compile(
    r"^(?:our|the company(?:'s|’s)?|tesla(?:'s|’s)?|ford(?:'s|’s)?|"
    r"delta(?:'s|’s)?|southwest(?:'s|’s)?)\s+(?:business|operations?|"
    r"subsidiar(?:y|ies)|refiner(?:y|ies)|facility|facilities|division|"
    r"network|stores?|service centers?|sales force|fleet|joint venture)",
    re.I,
)

DISCLOSURE_TERMS = re.compile(
    r"\b(?:annual report|quarterly report|current report|form 10-k|form 10-q|"
    r"form 8-k|sec filing|investor relations|available free of charge on our website|"
    r"corporate website|reports? filed with the sec|file with (?:the )?sec)\b",
    re.I,
)
ACCOUNTING_DISCLOSURE_TERMS = re.compile(
    r"\b(?:prepare segment results?|consolidate (?:that|financial|results?)|"
    r"reportable segments?|see in the united states and europe|present opportunities?)\b",
    re.I,
)
LEGAL_RISK_TERMS = re.compile(
    r"\b(?:face (?:the )?risk|risk of|lawsuit|litigation|legal proceeding|"
    r"comply with|compliance with|required to comply|subject to regulation|"
    r"regulatory requirement|government regulation|safety standard|epa standard|"
    r"nhtsa requirement|mount challenges|prevail in|request extensions? of .*authorit|"
    r"obtain .*government.*approval|obtain .*regulatory.*approval|"
    r"receive .*antitrust immunity|certify .*standard|meet all applicable .*standard|"
    r"subject to .*imposed by the faa)\b",
    re.I,
)
MARKET_CONDITION_TERMS = re.compile(
    r"^(?:face competition|compete for customers?|experience seasonal fluctuation|"
    r"experience demand fluctuation|prices? fluctuate|be subject to market condition)\b",
    re.I,
)
ASPIRATION_TERMS = re.compile(
    r"^(?:plan|intend|expect|believe|hope|aim|anticipate|seek|aspire)\b|"
    r"\b(?:we plan|we intend|we expect|we believe|our goal is|our objective is)\b",
    re.I,
)
OUTCOME_TERMS = re.compile(
    r"^(?:be|remain|become) (?:profitable|successful|competitive|a leader)|"
    r"^(?:earn|generate) (?:revenue|profit|income|credits?)\b|"
    r"^(?:achieve|deliver) (?:customer loyalty|high quality)\b",
    re.I,
)
RESOURCE_TERMS = re.compile(
    r"^(?:own|have|hold|possess)\b.*\b(?:patents?|trademarks?|service marks?|"
    r"facilities|plants?|employees?|aircraft|fleet|cash|brand|licenses?|stores?|"
    r"applications?)\b",
    re.I,
)
WEAK_STATE = re.compile(
    r"^(?:be|have|include|consist|represent|remain|seem|appear|face|experience|"
    r"constitute|comprise|depend|subject|consolidate|incur costs?|prepare segment|"
    r"present opportunities?|see in|designed? to allow)\b",
    re.I,
)
FRAGMENT_TERMS = re.compile(
    r"^(?:operate|serve|continue|use|provide|be)\s+(?:that|which|it|there|for)\b|"
    r"\b(?:that|which)\s+to\b$|^(?:receive for which|operate in the future)\b",
    re.I,
)
ONE_TIME_EVENT = re.compile(
    r"^(?:announce|acquire shares?|reach (?:a )?(?:tentative )?agreement|"
    r"incorporate|prevail|complete the acquisition|enter into a merger|"
    r"premiere .*campaign|introduce this program|begin customer deliveries|"
    r"complete significant investments?|add (?:its )?first .*destinations?)\b",
    re.I,
)
EXTRACTION_ARTIFACT = re.compile(
    r"^(?:sell|offer|incur|serve|premier)\s+(?:in addition|for the|during|in october)\b",
    re.I,
)


def canonical_hyphens(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    replacements = {
        r"\bpoint\s*-?\s*to\s*-?\s*point\b": "point-to-point",
        r"\bhub\s*-?\s*and\s*-?\s*spoke\b": "hub-and-spoke",
        r"\blow\s*-?\s*cost\b": "low-cost",
        r"\bsame\s*-?\s*day\b": "same-day",
        r"\bstore\s*-?\s*based\b": "store-based",
        r"\bin\s*-?\s*house\b": "in-house",
        r"\bself\s*-?\s*service\b": "self-service",
        r"\bcompany\s*-?\s*owned\b": "company-owned",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def normalize_phrase(text: str, nlp: Any) -> str:
    text = canonical_hyphens(str(text).lower().strip())
    doc = nlp(text)
    output: list[str] = []
    for token in doc:
        raw = token.text.lower().strip()
        if not raw or token.is_punct or raw in {"'s", "’s"}:
            continue
        if raw in PRESERVE_WORDS or "-" in raw:
            value = raw
        else:
            value = token.lemma_.lower().strip()
        if not value or value in GENERIC_STOPWORDS:
            continue
        output.append(value)
    cleaned = canonical_hyphens(" ".join(output))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def infer_actor_type(row: pd.Series, nlp: Any) -> str:
    source = re.sub(r"^\s*\d+\s+", "", str(row["source_sentence"]).strip())
    if re.search(
        r"\b(?:network|point-to-point|discount) carriers?.{0,180}\bprovide service at low fares\b",
        source,
        re.I,
    ):
        return "external_actor"
    # Attribute the sentence's grammatical subject before trusting extraction
    # metadata. This prevents a firm name elsewhere in a sentence from turning
    # an action by a union, regulator, customer, or supplier into a firm action.
    doc = nlp(source)
    roots = [token for token in doc if token.dep_ == "ROOT"]
    root_subjects = [
        token for token in doc
        if token.dep_ in {"nsubj", "nsubjpass"}
        and (token.head in roots or token.head.dep_ in {"conj", "xcomp", "ccomp"})
    ]
    subjects = root_subjects or [token for token in doc if token.dep_ in {"nsubj", "nsubjpass"}]
    for subject in subjects:
        subject_text = " ".join(token.text for token in subject.subtree).strip()
        if FIRM_UNIT_ACTOR.search(subject_text):
            return "firm_operating_unit"
        if FIRM_ACTOR.search(subject_text):
            return "firm"
        if AFFILIATE_ACTOR.search(subject_text):
            return "affiliate"
        if EXTERNAL_ACTOR.search(subject_text):
            return "external_actor"
        if PRODUCT_ACTOR.search(subject_text) or re.search(
            r"\b(?:powertrain|stations?|connectors?|facilities|network)\b",
            subject_text,
            re.I,
        ):
            return "product_or_system"
    recorded_subject = str(row.get("subject", "")).strip()
    if recorded_subject:
        if FIRM_UNIT_ACTOR.search(recorded_subject):
            return "firm_operating_unit"
        if FIRM_ACTOR.search(recorded_subject):
            return "firm"
        if AFFILIATE_ACTOR.search(recorded_subject):
            return "affiliate"
        if EXTERNAL_ACTOR.search(recorded_subject):
            return "external_actor"
        if PRODUCT_ACTOR.search(recorded_subject):
            return "product_or_system"
    if FIRM_UNIT_ACTOR.search(source):
        return "firm_operating_unit"
    if FIRM_ACTOR.search(source):
        return "firm"
    if AFFILIATE_ACTOR.search(source):
        return "affiliate"
    if re.search(r"\bby (?:us|the company|tesla|ford|delta|southwest)\b", source, re.I):
        return "implicit_or_passive_firm"
    return "unclear"


def content_word_count(cleaned: str, nlp: Any) -> int:
    return sum(
        token.pos_ in {"VERB", "NOUN", "PROPN", "ADJ", "ADV"}
        for token in nlp(cleaned)
    )


def classify_candidate(row: pd.Series, nlp: Any) -> dict[str, Any]:
    raw = str(row["activity_phrase_raw"])
    cleaned = str(row["activity_phrase_clean"])
    source = str(row["source_sentence"])
    combined = f"{cleaned} {source}"
    actor_type = infer_actor_type(row, nlp)
    tokens = cleaned.split()
    content_count = content_word_count(cleaned, nlp) if cleaned else 0
    granularity = "appropriate"
    construct = "operational_activity"
    reason = ""

    if len(tokens) <= 1:
        construct, reason, granularity = "fragment", "single_token_phrase", "fragment"
    elif not cleaned or content_count < 2:
        construct, reason, granularity = "fragment", "insufficient_content", "fragment"
    elif EXTRACTION_ARTIFACT.search(raw) or FRAGMENT_TERMS.search(cleaned) or re.search(r"\b(?:it|that|which)$", cleaned):
        construct, reason, granularity = "fragment", "syntactic_fragment", "fragment"
    elif DISCLOSURE_TERMS.search(combined):
        construct, reason = "disclosure", "website_or_disclosure_text"
    elif ACCOUNTING_DISCLOSURE_TERMS.search(combined):
        construct, reason = "disclosure", "accounting_or_segment_disclosure"
    elif LEGAL_RISK_TERMS.search(combined):
        construct, reason = "legal_or_risk", "legal_or_risk_disclosure"
    elif MARKET_CONDITION_TERMS.search(cleaned):
        construct, reason = "market_condition", "market_condition"
    elif ASPIRATION_TERMS.search(cleaned) or ASPIRATION_TERMS.search(source):
        construct, reason = "aspiration", "aspiration_or_future_intent"
    elif OUTCOME_TERMS.search(cleaned):
        construct, reason = "outcome", "outcome_not_activity"
    elif RESOURCE_TERMS.search(cleaned):
        construct, reason = "resource", "resource_not_activity"
    elif actor_type in {"external_actor", "product_or_system"}:
        construct, reason = "external_actor_action", "external_actor_action"
    elif actor_type == "unclear":
        construct, reason = "unattributed_action", "no_firm_actor_evidence"
    elif ONE_TIME_EVENT.search(cleaned):
        construct, reason, granularity = "one_time_event", "nonrepeatable_event", "too_narrow"
    elif WEAK_STATE.search(cleaned):
        construct, reason = "weak_state", "weak_verb_state"
    elif re.fullmatch(r"(?:provide|perform|conduct|manage|operate) (?:service|activity|operation|business)", cleaned):
        construct, reason, granularity = "overbroad_activity", "overbroad_generic_activity", "too_broad"

    return {
        "is_valid_candidate": reason == "",
        "drop_reason": reason,
        "construct_type": construct,
        "actor_type": actor_type,
        "granularity_flag": granularity,
    }


def protected_signature(phrase: str) -> tuple[str, ...]:
    phrase = canonical_hyphens(phrase.lower())
    signature = {modifier for modifier in PROTECTED_MODIFIERS if modifier in phrase}
    if re.search(r"\bthrough (?:dealers?|retailers?)\b", phrase):
        signature.add("through-dealers")
    if re.search(r"\bthrough (?:brokers?|agents?)\b", phrase):
        signature.add("through-brokers")
    if re.search(r"\bindependent (?:agents?|dealers?)\b", phrase):
        signature.add("independent-agent")
    if re.search(r"\bemployee salesforce\b", phrase):
        signature.add("employee-salesforce")
    return tuple(sorted(signature))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def near_duplicate_groups(
    frame: pd.DataFrame, embeddings: np.ndarray
) -> tuple[pd.DataFrame, list[str], list[int]]:
    size = len(frame)
    union_find = UnionFind(size)
    signatures = [protected_signature(value) for value in frame["activity_phrase_clean"]]
    similarity = embeddings @ embeddings.T
    for left in range(size):
        for right in range(left + 1, size):
            if similarity[left, right] >= SIMILARITY_THRESHOLD and signatures[left] == signatures[right]:
                union_find.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        components[union_find.find(index)].append(index)
    duplicate_components = [members for members in components.values() if len(members) > 1]
    duplicate_components.sort(key=lambda members: min(members))

    group_ids = [""] * size
    group_sizes = [1] * size
    summaries: list[dict[str, Any]] = []
    for number, members in enumerate(duplicate_components, start=1):
        group_id = f"NDM{number:04d}"
        for index in members:
            group_ids[index] = group_id
            group_sizes[index] = len(members)
        pair_scores = [
            float(similarity[a, b])
            for offset, a in enumerate(members)
            for b in members[offset + 1 :]
        ]
        subset = frame.iloc[members]
        summaries.append(
            {
                "near_duplicate_group_id": group_id,
                "group_size": len(members),
                "n_distinct_cleaned_phrases": subset["activity_phrase_clean"].nunique(),
                "firms": "|".join(sorted(subset["ticker"].astype(str).unique())),
                "source_methods": "|".join(sorted(subset["source_method"].astype(str).unique())),
                "protected_modifier_signature": "|".join(signatures[members[0]]),
                "min_pair_similarity": min(pair_scores),
                "max_pair_similarity": max(pair_scores),
                "member_cleaned_phrases": " || ".join(subset["activity_phrase_clean"].astype(str)),
                "member_source_sentences": " || ".join(subset["source_sentence"].astype(str)),
            }
        )
    return pd.DataFrame(summaries), group_ids, group_sizes


def add_summary(
    rows: list[dict[str, Any]], section: str, counts: dict[str, int] | Counter[str]
) -> None:
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append({"section": section, "category": key, "count": int(count)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--embedding-model", default=MODEL_NAME)
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-batch-size", type=int, default=20)
    parser.add_argument("--reuse-deepseek-audit", action="store_true")
    return parser.parse_args()


def deepseek_validity_audit(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Independently assess whether each provisionally valid activity is explicit and firm-performed."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for the validity reassessment")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.deepseek_base_url)
    records = frame.reset_index().rename(columns={"index": "row_id"})
    results: list[dict[str, Any]] = []
    system_prompt = (
        "You audit activities extracted from a firm's 10-K Item 1. Mark valid=true only when "
        "the source sentence explicitly states that the focal firm, its named subsidiary, or its "
        "clearly owned operating unit performs the concrete operational activity. Do not infer from "
        "industry knowledge. Mark false for actions by customers, employees, unions, suppliers, "
        "dealers, competitors, regulators, governments, products/systems acting autonomously, or an "
        "unidentified actor; and for risks, legal procedures, aspirations, outcomes, resources, states, "
        "or disclosure text. Passive voice is valid only if the firm is explicitly the agent or the "
        "firm's operation is unambiguous. Return JSON only: {\"assessments\":[{\"row_id\":integer,"
        "\"valid\":boolean,\"actor\":string,\"reason\":string}]}."
    )
    for start in range(0, len(records), args.deepseek_batch_size):
        batch = records.iloc[start:start + args.deepseek_batch_size]
        payload = [
            {
                "row_id": int(row.row_id), "firm": str(row.ticker),
                "activity_phrase": str(row.activity_phrase_raw),
                "source_sentence": str(row.source_sentence),
                "rule_actor_type": str(row.actor_type),
            }
            for row in batch.itertuples()
        ]
        last_error = None
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=args.deepseek_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"}, temperature=0,
                )
                parsed = json.loads(response.choices[0].message.content)
                assessed = parsed.get("assessments", [])
                by_id = {int(item["row_id"]): item for item in assessed}
                missing = [item["row_id"] for item in payload if item["row_id"] not in by_id]
                if missing:
                    raise ValueError(f"DeepSeek omitted row IDs: {missing}")
                for item in payload:
                    assessment = by_id[item["row_id"]]
                    results.append({
                        "row_id": item["row_id"],
                        "deepseek_valid": bool(assessment.get("valid", False)),
                        "deepseek_actor": str(assessment.get("actor", "")),
                        "deepseek_reason": str(assessment.get("reason", "")),
                        "deepseek_model": args.deepseek_model,
                        "deepseek_status": "ok",
                    })
                break
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    raise RuntimeError(f"DeepSeek validity batch failed at row {start}: {exc}") from exc
                time.sleep(2 ** attempt)
    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.comparison_dir.glob("*_medium_2015_comparison.csv"))
    if len(input_paths) != 4:
        raise SystemExit(f"Expected four medium comparison CSVs; found {len(input_paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in input_paths]
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(frame.columns.get_loc("activity_phrase") + 1, "activity_phrase_raw", frame["activity_phrase"])
    if "activity_phrase_clean" not in frame:
        frame["activity_phrase_clean"] = ""

    nlp = spacy.load(args.spacy_model, disable=["ner"])
    frame["activity_phrase_clean"] = [normalize_phrase(value, nlp) for value in frame["activity_phrase_raw"]]
    decisions = pd.DataFrame([classify_candidate(row, nlp) for _, row in frame.iterrows()])
    for column in decisions:
        frame[column] = decisions[column]

    # Preserve the deterministic decision, then apply an independent DeepSeek
    # review to every row. The final rule is conservative: both layers must
    # accept a phrase. This makes model decisions auditable and reproducible.
    frame["rule_is_valid_candidate"] = frame["is_valid_candidate"].astype(bool)
    frame["rule_drop_reason"] = frame["drop_reason"]
    audit_path = args.output_dir / "deepseek_validity_audit_medium.csv"
    if args.reuse_deepseek_audit and audit_path.exists():
        audit = pd.read_csv(audit_path)
        expected_ids = set(range(len(frame)))
        if set(audit["row_id"].astype(int)) != expected_ids:
            raise RuntimeError("Cached DeepSeek audit does not match current input rows")
        audit = audit[[
            "row_id", "deepseek_valid", "deepseek_actor", "deepseek_reason",
            "deepseek_model", "deepseek_status",
        ]].copy()
    else:
        audit = deepseek_validity_audit(frame, args)
        audit.to_csv(audit_path, index=False)
    frame = frame.reset_index().rename(columns={"index": "row_id"}).merge(
        audit, on="row_id", how="left", validate="one_to_one"
    )
    model_reject = frame["rule_is_valid_candidate"] & ~frame["deepseek_valid"]
    frame.loc[model_reject, "is_valid_candidate"] = False
    frame.loc[model_reject, "drop_reason"] = "deepseek_invalid_candidate"
    frame.loc[model_reject, "construct_type"] = "deepseek_rejected"

    # DeepSeek may resolve syntactically difficult actor attribution, but it
    # cannot override any substantive exclusion. Reclassify these rows with a
    # confirmed firm actor so later rules (one-time event, weak state, etc.)
    # still apply.
    actor_boundary = (
        frame["rule_drop_reason"].eq("no_firm_actor_evidence")
        & frame["deepseek_valid"].astype(bool)
    )
    for index in frame.index[actor_boundary]:
        review_row = frame.loc[index].copy()
        review_row["subject"] = "we"
        revised = classify_candidate(review_row, nlp)
        if revised["is_valid_candidate"]:
            revised["actor_type"] = "deepseek_confirmed_firm"
            revised["construct_type"] = "operational_activity"
            revised["drop_reason"] = ""
        for column, value in revised.items():
            frame.at[index, column] = value

    # Exact deduplication occurs after normalization and classification.
    key_columns = ["firm_id", "year", "activity_phrase_clean", "source_sentence"]
    duplicate_mask = frame.duplicated(key_columns, keep="first")
    frame.loc[duplicate_mask, "is_valid_candidate"] = False
    frame.loc[duplicate_mask, "drop_reason"] = "exact_duplicate"
    frame.loc[duplicate_mask, "construct_type"] = "duplicate"

    frame["deepseek_validation"] = np.where(
        frame["deepseek_valid"].astype(bool), "valid", "invalid"
    )
    frame["final_validation"] = np.where(
        frame["is_valid_candidate"].astype(bool), "valid", "invalid"
    )

    assessment_columns = [
        "row_id", "firm_id", "ticker", "year", "activity_phrase",
        "activity_phrase_raw",
        "activity_phrase_clean", "source_sentence", "source_method",
        "rule_is_valid_candidate", "rule_drop_reason", "actor_type",
        "deepseek_valid", "deepseek_validation", "deepseek_actor",
        "deepseek_reason", "is_valid_candidate", "final_validation",
        "drop_reason", "construct_type",
        "granularity_flag", "deepseek_model", "deepseek_status",
    ]
    combined_assessment = frame[assessment_columns].copy()
    combined_assessment.to_csv(
        args.output_dir / "validity_assessment_medium.csv", index=False
    )
    combined_assessment.to_csv(audit_path, index=False)
    combined_assessment.to_csv(
        args.output_dir / "activities_medium_cleaned_with_deepseek.csv",
        index=False,
    )

    valid = frame[frame["is_valid_candidate"]].copy().reset_index(drop=True)
    dropped = frame[~frame["is_valid_candidate"]].copy().reset_index(drop=True)
    valid.insert(0, "embedding_row", np.arange(len(valid), dtype=int))

    embedding_model = SentenceTransformer(args.embedding_model)
    embeddings = embedding_model.encode(
        valid["activity_phrase_clean"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("Embedding normalization check failed")

    near_summary, group_ids, group_sizes = near_duplicate_groups(valid, embeddings)
    valid["near_duplicate_group_id"] = group_ids
    valid["near_duplicate_group_size"] = group_sizes
    valid["embedding_model"] = args.embedding_model

    output_csv = args.output_dir / "activities_medium_cleaned.csv"
    valid.to_csv(output_csv, index=False)
    valid.to_parquet(args.output_dir / "activities_medium_cleaned.parquet", index=False)
    np.save(args.output_dir / "activity_embeddings_medium.npy", embeddings)
    dropped.to_csv(args.output_dir / "dropped_phrases_medium.csv", index=False)

    near_columns = [
        "near_duplicate_group_id", "group_size", "n_distinct_cleaned_phrases",
        "firms", "source_methods", "protected_modifier_signature",
        "min_pair_similarity", "max_pair_similarity", "member_cleaned_phrases",
        "member_source_sentences",
    ]
    if near_summary.empty:
        near_summary = pd.DataFrame(columns=near_columns)
    near_summary.to_csv(args.output_dir / "near_duplicate_summary_medium.csv", index=False)

    summary_rows: list[dict[str, Any]] = [
        {"section": "overall", "category": "raw_rows", "count": len(frame)},
        {"section": "overall", "category": "valid_candidates", "count": len(valid)},
        {"section": "overall", "category": "dropped_rows", "count": len(dropped)},
        {
            "section": "overall",
            "category": "exact_unique_cleaned_phrases",
            "count": valid["activity_phrase_clean"].nunique(),
        },
        {
            "section": "overall",
            "category": "near_duplicate_groups",
            "count": len(near_summary),
        },
    ]
    add_summary(summary_rows, "drop_reason", Counter(dropped["drop_reason"]))
    add_summary(summary_rows, "firm", Counter(valid["ticker"]))
    add_summary(summary_rows, "source_method", Counter(valid["source_method"]))
    add_summary(summary_rows, "construct_type_valid", Counter(valid["construct_type"]))
    add_summary(summary_rows, "actor_type_valid", Counter(valid["actor_type"]))
    add_summary(
        summary_rows,
        "deepseek_decision",
        Counter(frame["deepseek_valid"].map({True: "valid", False: "invalid"})),
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "cleaning_summary_medium.csv", index=False
    )

    print(f"Raw rows: {len(frame):,}")
    print(f"Valid candidates: {len(valid):,}")
    print(f"Dropped rows: {len(dropped):,}")
    print("Top drop reasons:")
    for reason, count in Counter(dropped["drop_reason"]).most_common(10):
        print(f"  {reason}: {count:,}")
    print(f"Exact unique cleaned phrases: {valid['activity_phrase_clean'].nunique():,}")
    print(f"Near-duplicate groups: {len(near_summary):,}")
    print("Valid candidates by firm:")
    for firm, count in Counter(valid["ticker"]).most_common():
        print(f"  {firm}: {count:,}")
    print("Valid candidates by source_method:")
    for method, count in Counter(valid["source_method"]).most_common():
        print(f"  {method}: {count:,}")


if __name__ == "__main__":
    main()
