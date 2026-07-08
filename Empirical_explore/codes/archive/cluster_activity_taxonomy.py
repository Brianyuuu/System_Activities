#!/usr/bin/env python3
"""Deduplicate, re-embed, cluster, and sanity-check cleaned activity phrases.

Inputs are the Step 3 medium cleaned activity set. Outputs are written to a
clustering folder and cover:

1. non-LLM duplicate detection and final deduplication;
2. optional DeepSeek duplicate audit when DEEPSEEK_API_KEY is set;
3. re-embedded final activity set;
4. Step 4 clustering runs and representative phrases;
5. Step 5 taxonomy sanity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sklearn.cluster import HDBSCAN
except Exception:  # pragma: no cover - depends on sklearn build.
    HDBSCAN = None  # type: ignore[assignment]


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NON_LLM_DUPLICATE_THRESHOLD = 0.92
STABILITY_SEEDS = [11, 23, 37, 49, 61]

PROTECTED_MODIFIERS = {
    "direct",
    "franchised",
    "electric",
    "low-cost",
    "premium",
    "point-to-point",
    "hub-and-spoke",
    "digital",
    "owned",
    "exclusive",
    "same-day",
    "store-based",
    "regional",
    "international",
    "online",
    "in-house",
    "outsourced",
    "domestic",
    "global",
    "company-owned",
    "through",
    "with",
    "by",
    "for",
    "under",
    "via",
    "using",
}

VALUE_CHAIN_PATTERNS = {
    "inbound_procurement_supply": re.compile(
        r"\b(procure|purchase|source|obtain|supplier|component|raw material|fuel|lease aircraft|finance)\b",
        re.I,
    ),
    "operations_manufacturing_service_delivery": re.compile(
        r"\b(manufactur|produc|assemble|operate|fly|serve|provide service|maintenance|repair|overhaul|machine|build|deliver|transport|schedule)\b",
        re.I,
    ),
    "outbound_distribution": re.compile(
        r"\b(distribut|deliver|ship|sell through|dealer|network|store|retail|route|destination|hub|point-to-point)\b",
        re.I,
    ),
    "marketing_sales_revenue": re.compile(
        r"\b(sell|market|advertis|price|fare|merchandis|reservation|loyalty|skymiles|rapid rewards|sales)\b",
        re.I,
    ),
    "customer_service_support": re.compile(
        r"\b(customer service|service center|support|warranty|repair|maintenance|call center|mobile technician|ranger)\b",
        re.I,
    ),
    "technology_r_and_d": re.compile(
        r"\b(research|r&d|develop|design|engineer|software|technology|innovation|test|battery|autopilot|navigation)\b",
        re.I,
    ),
    "partnerships_governance_coordination": re.compile(
        r"\b(joint venture|alliance|codeshare|partner|agreement|coordinate|arrangement|franchise|contract)\b",
        re.I,
    ),
}

GENERIC_CLUSTER_LABELS = {
    "manufactur": "manufacturing / production",
    "produc": "manufacturing / production",
    "assemble": "manufacturing / production",
    "sell": "sales / distribution",
    "market": "marketing / sales",
    "service": "customer service / support",
    "repair": "maintenance / repair",
    "maintain": "maintenance / repair",
    "operate": "operations / service delivery",
    "fly": "airline operations",
    "route": "network / route operations",
    "develop": "technology / product development",
    "design": "technology / product development",
    "test": "testing / R&D",
    "purchase": "procurement / sourcing",
    "procure": "procurement / sourcing",
    "source": "procurement / sourcing",
    "finance": "financing / leasing",
    "lease": "financing / leasing",
    "provide": "service provision",
    "joint": "alliances / partnerships",
    "codeshare": "alliances / partnerships",
    "fuel": "fuel / energy management",
    "loyalty": "loyalty programs",
    "reservation": "reservation / booking",
}


def canonical_hyphens(text: str) -> str:
    return (
        str(text)
        .replace("–", "-")
        .replace("—", "-")
        .replace("‑", "-")
        .lower()
        .strip()
    )


def protected_signature(text: str) -> str:
    phrase = canonical_hyphens(text)
    found = {word for word in PROTECTED_MODIFIERS if re.search(rf"\b{re.escape(word)}\b", phrase)}
    if re.search(r"\bthrough (?:dealers?|retailers?)\b", phrase):
        found.add("through-dealers")
    if re.search(r"\bthrough (?:brokers?|agents?)\b", phrase):
        found.add("through-brokers")
    if re.search(r"\bpoint-to-point\b", phrase):
        found.add("point-to-point")
    if re.search(r"\bhub-and-spoke\b", phrase):
        found.add("hub-and-spoke")
    return "|".join(sorted(found))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def choose_representative(group: pd.DataFrame) -> int:
    """Prefer cross-method rows, then LLM phrasing, then shorter clean phrase."""
    scored = group.copy()
    scored["_both"] = scored["source_method"].astype(str).str.contains("spacy_dependency") & scored[
        "source_method"
    ].astype(str).str.contains("deepseek")
    scored["_llm"] = scored["source_method"].astype(str).str.contains("deepseek")
    scored["_len"] = scored["activity_phrase_clean"].astype(str).str.len()
    scored = scored.sort_values(["_both", "_llm", "_len"], ascending=[False, False, True])
    return int(scored.index[0])


def aggregate_methods(values: pd.Series) -> str:
    joined = "|".join(values.dropna().astype(str))
    methods = []
    if "spacy_dependency" in joined:
        methods.append("spacy_dependency")
    if "deepseek_llm_chunk_extraction" in joined:
        methods.append("deepseek_llm_chunk_extraction")
    return "|".join(methods) if methods else joined


def method_match_from_methods(method: str) -> str:
    has_dep = "spacy_dependency" in str(method)
    has_llm = "deepseek_llm_chunk_extraction" in str(method)
    if has_dep and has_llm:
        return "both"
    if has_dep:
        return "dependency"
    if has_llm:
        return "llm"
    return "unknown"


def exact_clean_reconcile(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["firm_id", "year", "activity_phrase_clean", "source_sentence"]
    keep_rows: list[pd.Series] = []
    duplicate_rows: list[pd.DataFrame] = []
    for _, group in frame.groupby(key, dropna=False, sort=False):
        rep_idx = choose_representative(group)
        rep = frame.loc[rep_idx].copy()
        rep["source_method"] = aggregate_methods(group["source_method"])
        rep["method_match_clean"] = method_match_from_methods(rep["source_method"])
        rep["n_exact_clean_rows_collapsed"] = len(group)
        keep_rows.append(rep)
        dup = group.drop(index=rep_idx).copy()
        if not dup.empty:
            dup["duplicate_type"] = "exact_clean_phrase_and_source_sentence"
            dup["retained_activity_phrase_clean"] = rep["activity_phrase_clean"]
            duplicate_rows.append(dup)
    kept = pd.DataFrame(keep_rows).reset_index(drop=True)
    dropped = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else pd.DataFrame()
    return kept, dropped


def reconcile_methods_from_step3_dropped(valid: pd.DataFrame, dropped_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use Step 3 exact-duplicate dropped rows to relabel post-clean method overlap.

    Step 3 saved only the retained valid rows in activities_medium_cleaned.csv.
    Rows that normalized to the same clean phrase/source sentence were moved to
    dropped_phrases_medium.csv. Reading only the valid file therefore misses
    method overlap created by cleaning. This function uses dropped exact
    duplicates only as audit evidence; it does not restore them as activities.
    """
    frame = valid.copy()
    key = ["firm_id", "year", "activity_phrase_clean", "source_sentence"]
    if not dropped_path or not dropped_path.exists():
        frame["method_match_clean"] = frame["source_method"].map(method_match_from_methods)
        frame["n_exact_clean_rows_collapsed"] = frame.get("n_exact_clean_rows_collapsed", 1)
        return frame, pd.DataFrame()

    dropped = pd.read_csv(dropped_path, dtype=str, keep_default_na=False)
    exact = dropped[dropped.get("drop_reason", "").eq("exact_duplicate")].copy()
    if exact.empty:
        frame["method_match_clean"] = frame["source_method"].map(method_match_from_methods)
        frame["n_exact_clean_rows_collapsed"] = frame.get("n_exact_clean_rows_collapsed", 1)
        return frame, pd.DataFrame()

    audit_rows: list[dict[str, Any]] = []
    exact_groups = {tuple(row[col] for col in key): group for _, group in exact.groupby(key, dropna=False, sort=False) for row in [group.iloc[0]]}
    updated_methods = []
    method_matches = []
    collapsed_counts = []
    for _, row in frame.iterrows():
        k = tuple(row[col] for col in key)
        group = exact_groups.get(k)
        if group is None:
            method = str(row.get("source_method", ""))
            collapsed = int(row.get("n_exact_clean_rows_collapsed", 1) or 1)
        else:
            method = aggregate_methods(pd.concat([pd.Series([row.get("source_method", "")]), group["source_method"]]))
            collapsed = 1 + len(group)
            audit_rows.append(
                {
                    "firm_id": row.get("firm_id", ""),
                    "ticker": row.get("ticker", ""),
                    "year": row.get("year", ""),
                    "activity_phrase_clean": row.get("activity_phrase_clean", ""),
                    "source_sentence": row.get("source_sentence", ""),
                    "retained_source_method_before": row.get("source_method", ""),
                    "source_method_after_clean_reconciliation": method,
                    "method_match_clean": method_match_from_methods(method),
                    "n_step3_exact_duplicates": len(group),
                    "dropped_activity_phrases": " || ".join(group["activity_phrase_raw"].astype(str)),
                    "dropped_source_methods": " || ".join(group["source_method"].astype(str)),
                }
            )
        updated_methods.append(method)
        method_matches.append(method_match_from_methods(method))
        collapsed_counts.append(collapsed)
    frame["source_method"] = updated_methods
    frame["method_match_clean"] = method_matches
    frame["n_exact_clean_rows_collapsed"] = collapsed_counts
    return frame, pd.DataFrame(audit_rows)


def non_llm_semantic_deduplicate(
    frame: pd.DataFrame, embeddings: np.ndarray, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    signatures = frame["activity_phrase_clean"].map(protected_signature).tolist()
    similarity = embeddings @ embeddings.T
    uf = UnionFind(n)
    candidate_pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if signatures[i] != signatures[j]:
                continue
            if similarity[i, j] < threshold:
                continue
            # Keep scope-specific phrases separate when source sentence differs
            # and one phrase contains another with extra strategic mode words.
            left = str(frame.loc[i, "activity_phrase_clean"])
            right = str(frame.loc[j, "activity_phrase_clean"])
            if left != right and {left, right} and (left in right or right in left):
                extra = right.replace(left, "") if left in right else left.replace(right, "")
                if protected_signature(extra):
                    continue
            uf.union(i, j)
            candidate_pairs.append(
                {
                    "left_embedding_row": i,
                    "right_embedding_row": j,
                    "left_phrase": left,
                    "right_phrase": right,
                    "left_firm": frame.loc[i, "ticker"],
                    "right_firm": frame.loc[j, "ticker"],
                    "cosine_similarity": float(similarity[i, j]),
                    "protected_modifier_signature": signatures[i],
                }
            )

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)
    groups = [members for members in components.values() if len(members) > 1]
    groups.sort(key=min)

    group_id = [""] * n
    rows: list[dict[str, Any]] = []
    keep_indices: list[int] = []
    semantic_dropped: list[pd.DataFrame] = []
    for number, members in enumerate(groups, start=1):
        gid = f"DEDUP{number:04d}"
        subset = frame.iloc[members].copy()
        rep_pos = choose_representative(subset)
        rep_loc = frame.index.get_loc(rep_pos)
        for member in members:
            group_id[member] = gid
        keep_indices.append(rep_pos)
        dropped = subset.drop(index=rep_pos).copy()
        if not dropped.empty:
            dropped["duplicate_type"] = "non_llm_semantic_near_duplicate"
            dropped["non_llm_duplicate_group_id"] = gid
            dropped["retained_activity_phrase_clean"] = frame.loc[rep_pos, "activity_phrase_clean"]
            semantic_dropped.append(dropped)
        pair_scores = [similarity[a, b] for a, b in combinations(members, 2)]
        rows.append(
            {
                "non_llm_duplicate_group_id": gid,
                "group_size": len(members),
                "retained_embedding_row": rep_loc,
                "retained_phrase": frame.loc[rep_pos, "activity_phrase_clean"],
                "firms": "|".join(sorted(subset["ticker"].astype(str).unique())),
                "source_methods": "|".join(sorted(subset["source_method"].astype(str).unique())),
                "min_pair_similarity": float(np.min(pair_scores)),
                "max_pair_similarity": float(np.max(pair_scores)),
                "member_phrases": " || ".join(subset["activity_phrase_clean"].astype(str)),
                "member_source_sentences": " || ".join(subset["source_sentence"].astype(str)),
            }
        )

    all_group_members = {member for group in groups for member in group}
    keep_from_groups = set(keep_indices)
    keep_mask = [
        (i not in all_group_members) or (frame.index[i] in keep_from_groups)
        for i in range(n)
    ]
    final = frame.loc[keep_mask].copy().reset_index(drop=True)
    final["non_llm_duplicate_group_id"] = [
        group_id[i] for i, keep in enumerate(keep_mask) if keep
    ]
    dropped_semantic = (
        pd.concat(semantic_dropped, ignore_index=True) if semantic_dropped else pd.DataFrame()
    )
    return final, pd.DataFrame(rows), pd.DataFrame(candidate_pairs), dropped_semantic


def embed_phrases(phrases: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        phrases,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def top_terms_for_subset(phrases: list[str], n: int = 6) -> str:
    if not phrases:
        return ""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(phrases)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    names = np.array(vectorizer.get_feature_names_out())
    top = names[np.argsort(scores)[::-1][:n]]
    return ", ".join(top.tolist())


def heuristic_label(phrases: list[str]) -> str:
    text = " ".join(phrases).lower()
    hits = Counter()
    for needle, label in GENERIC_CLUSTER_LABELS.items():
        if needle in text:
            hits[label] += text.count(needle)
    if hits:
        return hits.most_common(1)[0][0]
    terms = top_terms_for_subset(phrases, 3)
    return terms or "miscellaneous activities"


def representative_rows(
    frame: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    reps: list[dict[str, Any]] = []
    for label in sorted(set(labels)):
        subset_idx = np.where(labels == label)[0]
        subset = frame.iloc[subset_idx]
        phrases = subset["activity_phrase_clean"].astype(str).tolist()
        if label == -1:
            centroid = embeddings[subset_idx].mean(axis=0)
            readable = "noise / unclustered"
        else:
            centroid = embeddings[subset_idx].mean(axis=0)
            readable = heuristic_label(phrases)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = embeddings[subset_idx] @ centroid
        order = np.argsort(sims)[::-1][:10]
        cluster_id = f"{method}_{int(label)}"
        firms = sorted(subset["ticker"].astype(str).unique())
        rows.append(
            {
                "method": method,
                "cluster_label": int(label),
                "cluster_id": cluster_id,
                "human_readable_label": readable,
                "cluster_size": len(subset),
                "n_firms": len(firms),
                "firms": "|".join(firms),
                "top_terms": top_terms_for_subset(phrases, 8),
                "industry_scope": "industry_spanning" if len(firms) >= 3 else "industry_specific_or_firm_specific",
            }
        )
        for rank, local_pos in enumerate(order, start=1):
            row = subset.iloc[int(local_pos)]
            reps.append(
                {
                    "method": method,
                    "cluster_label": int(label),
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "similarity_to_centroid": float(sims[int(local_pos)]),
                    "ticker": row.get("ticker", ""),
                    "firm_id": row.get("firm_id", ""),
                    "activity_phrase_clean": row.get("activity_phrase_clean", ""),
                    "activity_phrase_raw": row.get("activity_phrase_raw", ""),
                    "source_sentence": row.get("source_sentence", ""),
                    "source_method": row.get("source_method", ""),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(reps)


def run_clusterings(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    representatives: list[pd.DataFrame] = []

    def add_labels(method: str, labels: np.ndarray) -> None:
        label_frames.append(
            pd.DataFrame(
                {
                    "activity_id": frame["activity_id"],
                    "method": method,
                    "cluster_label": labels.astype(int),
                    "cluster_id": [f"{method}_{int(x)}" for x in labels],
                }
            )
        )
        summary, reps = representative_rows(frame, embeddings, labels, method)
        summaries.append(summary)
        representatives.append(reps)

    if HDBSCAN is not None:
        for min_cluster_size, min_samples in [(5, 2), (8, 3), (12, 4)]:
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = model.fit_predict(embeddings)
            add_labels(f"hdbscan_mcs{min_cluster_size}_ms{min_samples}", labels)

    for k in [12, 18, 24]:
        model = KMeans(n_clusters=k, n_init=25, random_state=42)
        labels = model.fit_predict(embeddings)
        add_labels(f"kmeans_k{k}", labels)

    for k in [12, 18, 24]:
        distance = pairwise_distances(embeddings, metric="cosine")
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(distance)
        add_labels(f"hierarchical_avg_k{k}", labels)

    # BERTopic-style proxy: use embedding clusters plus c-TF-IDF/top phrases.
    # This is recorded explicitly as a proxy because BERTopic is not installed.
    for k in [18]:
        model = KMeans(n_clusters=k, n_init=25, random_state=314)
        labels = model.fit_predict(embeddings)
        add_labels(f"bertopic_proxy_k{k}", labels)

    return (
        pd.concat(label_frames, ignore_index=True),
        pd.concat(summaries, ignore_index=True),
        pd.concat(representatives, ignore_index=True),
    )


def clustering_stability(embeddings: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in [12, 18, 24]:
        labels_by_seed = []
        for seed in STABILITY_SEEDS:
            labels_by_seed.append(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings))
        scores = [
            adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
            for i, j in combinations(range(len(labels_by_seed)), 2)
        ]
        rows.append(
            {
                "method_family": "kmeans",
                "parameter": f"k={k}",
                "n_runs": len(labels_by_seed),
                "mean_adjusted_rand_index": float(np.mean(scores)),
                "min_adjusted_rand_index": float(np.min(scores)),
                "max_adjusted_rand_index": float(np.max(scores)),
            }
        )
    return pd.DataFrame(rows)


def sanity_checks(frame: pd.DataFrame, cluster_summary: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for category, pattern in VALUE_CHAIN_PATTERNS.items():
        mask = frame["activity_phrase_clean"].astype(str).str.contains(pattern)
        rows.append(
            {
                "sanity_check": "value_chain_recovery",
                "category": category,
                "n_matching_activities": int(mask.sum()),
                "n_matching_firms": int(frame.loc[mask, "ticker"].nunique()),
                "example_phrases": " || ".join(frame.loc[mask, "activity_phrase_clean"].head(8).astype(str)),
            }
        )
    scope = cluster_summary.groupby(["method", "industry_scope"]).size().reset_index(name="n_clusters")
    return pd.DataFrame(rows), scope


def deepseek_duplicate_audit(
    frame: pd.DataFrame,
    non_llm_pairs: pd.DataFrame,
    output_dir: Path,
    model: str,
    base_url: str,
    max_pairs: int,
) -> pd.DataFrame:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "DEEPSEEK_API_KEY not set",
                    "model": model,
                    "n_pairs_reviewed": 0,
                }
            ]
        )
        skipped.to_csv(output_dir / "deepseek_duplicate_audit.csv", index=False)
        return skipped

    pairs = non_llm_pairs.sort_values("cosine_similarity", ascending=False).head(max_pairs).copy()
    client = OpenAI(api_key=api_key, base_url=base_url)
    results = []
    system = (
        "You audit duplicate operational activity phrases from 10-K Item 1. "
        "Return only JSON. A duplicate means both phrases describe the same firm-performed operational activity. "
        "Do not merge phrases that differ in strategic mode/scope, such as direct vs dealer, "
        "owned vs outsourced, point-to-point vs hub-and-spoke, domestic vs international."
    )
    for _, row in pairs.iterrows():
        user = {
            "left_phrase": row["left_phrase"],
            "right_phrase": row["right_phrase"],
            "left_firm": row["left_firm"],
            "right_firm": row["right_firm"],
            "cosine_similarity": row["cosine_similarity"],
            "instruction": "Return JSON with fields is_duplicate (boolean), confidence (0-1), reason (short).",
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            results.append({**row.to_dict(), **parsed, "status": "ok", "model": model})
            time.sleep(0.05)
        except Exception as exc:
            results.append({**row.to_dict(), "status": "error", "error": str(exc), "model": model})
    audit = pd.DataFrame(results)
    audit.to_csv(output_dir / "deepseek_duplicate_audit.csv", index=False)
    return audit


def compare_dedup_methods(non_llm_pairs: pd.DataFrame, deepseek_audit: pd.DataFrame) -> pd.DataFrame:
    if deepseek_audit.empty or "is_duplicate" not in deepseek_audit:
        return pd.DataFrame(
            [
                {
                    "comparison": "non_llm_vs_deepseek",
                    "status": "deepseek_unavailable",
                    "n_non_llm_duplicate_pairs": len(non_llm_pairs),
                    "n_deepseek_reviewed_pairs": 0,
                }
            ]
        )
    audited = deepseek_audit[deepseek_audit["status"].eq("ok")].copy()
    audited["deepseek_duplicate_bool"] = audited["is_duplicate"].astype(str).str.lower().isin(["true", "1", "yes"])
    return pd.DataFrame(
        [
            {
                "comparison": "non_llm_vs_deepseek",
                "status": "completed",
                "n_non_llm_duplicate_pairs": len(non_llm_pairs),
                "n_deepseek_reviewed_pairs": len(audited),
                "n_deepseek_agrees_duplicate": int(audited["deepseek_duplicate_bool"].sum()),
                "n_deepseek_rejects_duplicate": int((~audited["deepseek_duplicate_bool"]).sum()),
                "deepseek_agreement_rate_on_reviewed": float(audited["deepseek_duplicate_bool"].mean()) if len(audited) else np.nan,
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dropped-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--duplicate-threshold", type=float, default=NON_LLM_DUPLICATE_THRESHOLD)
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-max-pairs", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    raw["protected_modifier_signature"] = raw["activity_phrase_clean"].map(protected_signature)

    exact_reconciled, exact_dropped = exact_clean_reconcile(raw)
    exact_reconciled, step3_method_audit = reconcile_methods_from_step3_dropped(
        exact_reconciled, args.dropped_input
    )
    exact_reconciled["activity_id"] = [f"ACT{i:05d}" for i in range(len(exact_reconciled))]

    initial_embeddings = embed_phrases(
        exact_reconciled["activity_phrase_clean"].astype(str).tolist(), args.embedding_model
    )
    final, non_llm_groups, non_llm_pairs, semantic_dropped = non_llm_semantic_deduplicate(
        exact_reconciled, initial_embeddings, args.duplicate_threshold
    )
    final = final.reset_index(drop=True)
    final["activity_id"] = [f"ACT{i:05d}" for i in range(len(final))]
    final_embeddings = embed_phrases(final["activity_phrase_clean"].astype(str).tolist(), args.embedding_model)
    final["embedding_model"] = args.embedding_model

    final.to_csv(args.output_dir / "activities_medium_final_deduped.csv", index=False)
    final.to_parquet(args.output_dir / "activities_medium_final_deduped.parquet", index=False)
    np.save(args.output_dir / "activity_embeddings_medium_final.npy", final_embeddings)
    exact_dropped.to_csv(args.output_dir / "exact_clean_duplicates_dropped.csv", index=False)
    step3_method_audit.to_csv(args.output_dir / "post_clean_method_reconciliation.csv", index=False)
    semantic_dropped.to_csv(args.output_dir / "non_llm_semantic_duplicates_dropped.csv", index=False)
    non_llm_groups.to_csv(args.output_dir / "non_llm_duplicate_groups.csv", index=False)
    non_llm_pairs.to_csv(args.output_dir / "non_llm_duplicate_pairs.csv", index=False)

    deepseek_audit = deepseek_duplicate_audit(
        final,
        non_llm_pairs,
        args.output_dir,
        args.deepseek_model,
        args.deepseek_base_url,
        args.deepseek_max_pairs,
    )
    dedup_comparison = compare_dedup_methods(non_llm_pairs, deepseek_audit)
    dedup_comparison.to_csv(args.output_dir / "duplicate_method_comparison.csv", index=False)

    labels, cluster_summary, representatives = run_clusterings(final, final_embeddings)
    labels.to_csv(args.output_dir / "activity_cluster_assignments.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    representatives.to_csv(args.output_dir / "cluster_representative_phrases.csv", index=False)
    clustering_stability(final_embeddings).to_csv(args.output_dir / "cluster_stability.csv", index=False)

    value_chain, scope = sanity_checks(final, cluster_summary, labels)
    value_chain.to_csv(args.output_dir / "taxonomy_value_chain_sanity_check.csv", index=False)
    scope.to_csv(args.output_dir / "taxonomy_industry_scope_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "input_cleaned_rows", "value": len(raw)},
            {"metric": "after_exact_clean_reconciliation", "value": len(exact_reconciled)},
            {"metric": "exact_clean_rows_dropped", "value": len(exact_dropped)},
            {"metric": "post_clean_method_reconciled_rows", "value": len(step3_method_audit)},
            {
                "metric": "post_clean_both_method_rows",
                "value": int((exact_reconciled["method_match_clean"] == "both").sum()),
            },
            {"metric": "non_llm_duplicate_groups", "value": len(non_llm_groups)},
            {"metric": "non_llm_duplicate_pairs", "value": len(non_llm_pairs)},
            {"metric": "semantic_rows_dropped_non_llm", "value": len(semantic_dropped)},
            {"metric": "final_deduped_rows", "value": len(final)},
            {"metric": "cluster_assignment_rows", "value": len(labels)},
            {"metric": "clustering_methods", "value": labels["method"].nunique()},
            {
                "metric": "deepseek_status",
                "value": deepseek_audit.iloc[0]["status"] if len(deepseek_audit) else "not_run",
            },
        ]
    )
    summary.to_csv(args.output_dir / "clustering_pipeline_summary.csv", index=False)

    narrative = [
        "Step 4/5 notes",
        "",
        "Input is the medium cleaned activity set. Exact clean duplicates are first reconciled so source_method and method_match_clean reflect dependency, LLM, or both after cleaning.",
        f"Non-LLM near-duplicate detection uses normalized all-mpnet-base-v2 embeddings, cosine >= {args.duplicate_threshold}, and protected strategic modifier signatures to avoid merging strategic-mode differences.",
        "Final activities are re-embedded after deduplication, then clustered with HDBSCAN where available, k-means, hierarchical average-linkage clustering, and a BERTopic-style proxy because BERTopic is not installed in this environment.",
        "Cluster labels are heuristic labels from representative phrase terms; they are meant for human review, not final theory coding.",
        "Step 5 sanity checks report whether value-chain-like categories are present and whether clusters are industry-spanning or firm/industry-specific.",
        "",
        "DeepSeek duplicate audit status: "
        + (str(deepseek_audit.iloc[0].get("status")) if len(deepseek_audit) else "not_run"),
    ]
    (args.output_dir / "step4_step5_narrative.md").write_text("\n".join(narrative), encoding="utf-8")

    print(f"Input cleaned rows: {len(raw):,}")
    print(f"After exact clean reconciliation: {len(exact_reconciled):,}")
    print(f"Exact clean rows dropped: {len(exact_dropped):,}")
    print(f"Post-clean method reconciled rows: {len(step3_method_audit):,}")
    print(f"Post-clean both-method rows: {(exact_reconciled['method_match_clean'] == 'both').sum():,}")
    print(f"Non-LLM duplicate groups: {len(non_llm_groups):,}")
    print(f"Semantic rows dropped by non-LLM dedup: {len(semantic_dropped):,}")
    print(f"Final deduped activities: {len(final):,}")
    print(f"Clustering methods: {labels['method'].nunique():,}")
    print(f"DeepSeek audit status: {deepseek_audit.iloc[0]['status'] if len(deepseek_audit) else 'not_run'}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
