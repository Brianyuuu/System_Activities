"""Run the frozen v5.8 phrase workflow with cross-industry prompts on PARCC."""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv,
    resolve_project_path,
    sha256_file,
    utc_now,
)


FINAL_FIELDS = [
    "ticker",
    "year",
    "atom_mechanism_phrase",
    "activity_phrase",
    "original_sentence",
]
SPECIFICITY_AUDIT_FIELDS = FINAL_FIELDS + [
    "specificity_status",
    "specificity_rule",
    "specificity_normalized_phrase",
]

PHRASE_PIPELINE_VERSION = "historical-v5.8-specificity-v1-2026-09-16"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _run(command: list[str], *, env: Mapping[str, str], logger: logging.Logger) -> None:
    script_name = Path(command[1]).name
    repair_start = int(env.get("PHRASE_REPAIR_START_STAGE", "1"))
    stage_number = {
        "01b_context_clausal_actor_extraction_v5_5.py": 1,
        "02_filter_cross_industry_focal.py": 2,
        "03_2_audit_candidate_mechanism_type_v5_3.py": 3,
        "04_prepare_unmodified_mechanism_input.py": 4,
        "05_decompose_mechanism_phrases.py": 5,
    }.get(script_name)
    if stage_number is not None and stage_number < repair_start:
        logger.info("Reusing verified prior stage %s for repair", script_name)
        return
    logger.info("Latest phrase stage: %s", script_name)
    subprocess.run(command, check=True, env=dict(env))


def _api_environment(settings: Mapping[str, Any], project_root: Path) -> dict[str, str]:
    env_file = resolve_project_path(project_root, settings.get("env_file", ".env"))
    _load_dotenv(env_file)
    key_name = str(settings.get("api_key_env", "LITELLM_API_KEY"))
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        raise RuntimeError(f"{key_name} is not set; checked process environment and {env_file}")
    base_name = str(settings.get("base_url_env", "LITELLM_BASE_URL"))
    base_url = os.environ.get(base_name, str(settings.get("base_url", "https://litellm.parcc.upenn.edu"))).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    env = os.environ.copy()
    # The frozen scripts use DeepSeek-compatible environment names.  Only the
    # transport is adapted; the frozen prompts, schemas, and stage order remain unchanged.
    env["DEEPSEEK_API_KEY"] = api_key
    env["DEEPSEEK_BASE_URL"] = base_url
    return env


def _prepare_input(
    filing: Mapping[str, str], filing_root: Path, project_root: Path, logger: logging.Logger
) -> tuple[Path, Path]:
    input_dir = filing_root / "_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    accession = filing["accession_number"]
    raw_path = Path(filing["raw_path"])
    if not raw_path.is_file():
        local_sample = project_root / "local_view" / "raw_samples" / f"{accession}.txt"
        if local_sample.is_file():
            raw_path = local_sample
        else:
            raise FileNotFoundError(f"raw filing is missing: {raw_path}")
    input_file = input_dir / f"{accession}.txt"
    if input_file.is_symlink() and input_file.resolve() != raw_path.resolve():
        input_file.unlink()
    if not input_file.exists():
        input_file.symlink_to(raw_path.resolve())
    clean_item1 = Path(str(filing.get("item1_path", "")))
    fallback_file = input_dir / f"{accession}_clean_item1.md"
    if not clean_item1.is_file():
        raise FileNotFoundError(f"canonical Item 1 is missing: {clean_item1}")
    expected_item1_hash = str(filing.get("item1_sha256", "")).strip()
    actual_item1_hash = sha256_file(clean_item1)
    if not expected_item1_hash or actual_item1_hash != expected_item1_hash:
        raise RuntimeError(
            f"canonical Item 1 hash mismatch for {accession}: "
            f"manifest={expected_item1_hash!r} actual={actual_item1_hash}"
        )
    if fallback_file.is_symlink() and fallback_file.resolve() != clean_item1.resolve():
        fallback_file.unlink()
    if not fallback_file.exists():
        fallback_file.symlink_to(clean_item1.resolve())
    fallback_name = fallback_file.name
    manifest = filing_root / "run_manifest.csv"
    atomic_write_csv(
        manifest,
        [
            {
                "sample_id": filing["sample_id"],
                "ticker": filing.get("ticker", "") or filing["sample_id"],
                "company": filing.get("company_name_wrds", "") or filing.get("ticker", "") or filing["sample_id"],
                "report_year": filing["target_fiscal_year"],
                "accession_number": accession,
                "filing_date": filing.get("filing_date", ""),
                "form": filing.get("form", "10-K"),
                "input_file": input_file.name,
                "clean_item1_file": fallback_name,
            }
        ],
    )
    logger.info("Prepared latest-pipeline manifest for %s", filing["sample_id"])
    return input_dir, manifest


def _finish_zero(
    filing_root: Path, filing: Mapping[str, str], reason: str,
    model: str, audit_model: str,
) -> dict[str, Any]:
    final_dir = filing_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "atom_mechanism_phrases_03_2.csv"
    atomic_write_csv(final_path, [], FINAL_FIELDS)
    specificity_dir = filing_root / "12_specificity_audit"
    specificity_audit = specificity_dir / "phrase_specificity_audit.csv"
    atomic_write_csv(specificity_audit, [], SPECIFICITY_AUDIT_FIELDS)
    summary = {
        "status": "pass",
        "pipeline_version": PHRASE_PIPELINE_VERSION,
        "extraction_model": model,
        "audit_model": audit_model,
        "sample_id": filing["sample_id"],
        "year": filing["target_fiscal_year"],
        "accession": filing["accession_number"],
        "item1_sha256": filing.get("item1_sha256", ""),
        "specificity_audit_input": 0,
        "specificity_audit_rejected": 0,
        "specificity_audit_output": str(specificity_audit.resolve()),
        "final_rows": 0,
        "reason": reason,
        "final_output": str(final_path.resolve()),
    }
    atomic_write_json(filing_root / "latest_pipeline_summary.json", summary)
    return summary


def _run_filing(
    filing: Mapping[str, str], settings: Mapping[str, Any], project_root: Path,
    env: Mapping[str, str], logger: logging.Logger,
) -> dict[str, Any]:
    output_root = resolve_project_path(project_root, settings.get("output_root", "outputs/phrases_latest"))
    filing_root = output_root / filing["sample_id"] / filing["target_fiscal_year"] / filing["accession_number"]
    filing_root.mkdir(parents=True, exist_ok=True)
    repair_start = int(os.getenv("PHRASE_REPAIR_START_STAGE", "1"))
    if repair_start not in {1, 5, 6}:
        raise ValueError("PHRASE_REPAIR_START_STAGE must be 1, 5, or 6")
    if repair_start > 1:
        prior_summary_path = filing_root / "latest_pipeline_summary.json"
        if not prior_summary_path.is_file():
            raise RuntimeError("Stage repair requires an existing filing summary")
        prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        if (
            prior_summary.get("status") != "pass"
            or prior_summary.get("sample_id") != filing["sample_id"]
            or prior_summary.get("year") != filing["target_fiscal_year"]
            or prior_summary.get("accession") != filing["accession_number"]
            or prior_summary.get("item1_sha256") != filing.get("item1_sha256", "")
        ):
            raise RuntimeError("Stage repair prerequisite summary does not match the current filing")
        required_prior = [
            filing_root / "01_extraction" / "extraction_audit.csv",
            filing_root / "02_actor_filter" / "curated_candidates.csv",
            filing_root / "03_2_candidate_audit" / "audited_keep.csv",
            filing_root / "04_unmodified.csv",
        ]
        if repair_start == 6:
            required_prior.append(filing_root / "05_decomposed.csv")
        missing_prior = [str(path) for path in required_prior if not path.is_file()]
        if missing_prior:
            raise RuntimeError(f"Stage repair prerequisites are missing: {missing_prior}")
    input_dir, manifest = _prepare_input(filing, filing_root, project_root, logger)
    source = resolve_project_path(project_root, settings.get("source_root", "pipeline_latest/codes"))
    prompts = resolve_project_path(project_root, settings.get("prompt_root", "pipeline_latest/prompts_generic"))
    python = sys.executable
    model = str(settings.get("model", "openai/gpt-oss-20b"))
    audit_model = str(settings.get("audit_model", model))
    worker_setting = os.getenv("PHRASE_API_WORKERS", str(settings.get("api_workers", 128)))
    workers = str(max(1, int(worker_setting)))
    retries = str(max(0, int(settings.get("max_retries", 2))))
    year = filing["target_fiscal_year"]
    stage01 = filing_root / "01_extraction"

    _run(
        [
            python, str(source / "01b_context_clausal_actor_extraction_v5_5.py"),
            "--input-dir", str(input_dir), "--manifest-csv", str(manifest),
            "--output-dir", str(stage01),
            "--prompt-file", str(prompts / "01_actor_context_clausal_v5_3.txt"),
            "--repair-prompt-file", str(prompts / "01_actor_context_clausal_repair_v5_3.txt"),
            "--firms", "ALL", "--min-year", year, "--max-year", year,
            "--chunk-chars", str(settings.get("chunk_chars", 3000)),
            "--clause-threshold", str(settings.get("clause_threshold", 350)),
            "--item1-min-words", str(settings.get("item1_min_words", 500)),
            "--item1-warning-words", str(settings.get("item1_warning_words", 1500)),
            "--model", model, "--api-workers", workers, "--submission-batch-size", workers,
            "--cpu-workers", str(settings.get("cpu_workers", 4)),
            "--max-tokens", str(settings.get("max_tokens", 24000)), "--max-retries", retries,
            "--allow-boundary-warnings", "--bisect-on-failure",
            "--fallback-to-source-on-failure",
        ] + ([] if model == "parcc/GLM" else ["--cache-optimized-prompt-format"]),
        env=env,
        logger=logger,
    )

    stage02 = filing_root / "02_actor_filter"
    _run(
        [
            python, str(project_root / "pipeline_latest" / "02_filter_cross_industry_focal.py"),
            "--input-csv", str(stage01 / "keep.csv"), "--output-dir", str(stage02),
            "--ticker", filing.get("ticker", "") or filing["sample_id"],
            "--company", filing.get("company_name_wrds", "") or filing.get("ticker", "") or filing["sample_id"],
        ],
        env=env,
        logger=logger,
    )
    curated = stage02 / "curated_candidates.csv"
    if _count_rows(curated) == 0:
        return _finish_zero(filing_root, filing, "no_focal_candidates", model, audit_model)

    stage03 = filing_root / "03_2_candidate_audit"
    _run(
        [
            python, str(source / "03_2_audit_candidate_mechanism_type_v5_3.py"),
            "--input-csv", str(curated), "--output-dir", str(stage03),
            "--prompt-file", str(prompts / "03_2_candidate_mechanism_type_v5_3.txt"),
            "--model", audit_model, "--api-base-url", env["DEEPSEEK_BASE_URL"],
            "--api-key-env", "DEEPSEEK_API_KEY", "--api-workers", workers,
            "--batch-size", "8",
            "--max-retries", str(settings.get("audit_max_retries", 3)),
            "--request-timeout", str(settings.get("audit_request_timeout", 90)),
            "--max-tokens", str(settings.get("audit_max_tokens", 4000)),
            "--bisect-on-failure", "--fallback-to-reject-on-failure",
        ],
        env=env,
        logger=logger,
    )
    audited = stage03 / "audited_keep.csv"
    if _count_rows(audited) == 0:
        return _finish_zero(filing_root, filing, "all_candidates_rejected", model, audit_model)

    unmodified = filing_root / "04_unmodified.csv"
    _run([python, str(source / "04_prepare_unmodified_mechanism_input.py"), "--input", str(audited), "--output", str(unmodified)], env=env, logger=logger)
    decomposed = filing_root / "05_decomposed.csv"
    _run(
        [
            python, str(source / "05_decompose_mechanism_phrases.py"),
            "--input", str(unmodified), "--output", str(decomposed),
            "--system-prompt-file", str(prompts / "05_decompose_mechanism_phrases_generic.txt"),
            "--model", model, "--base-url", env["DEEPSEEK_BASE_URL"],
            "--api-key-env", "DEEPSEEK_API_KEY", "--api-workers", workers,
            "--batch-size", str(settings.get("decompose_batch_size", 4)),
            "--max-retries", str(settings.get("decompose_max_retries", 1)),
            "--fallback-to-source-on-failure",
            "--no-resume",
        ],
        env=env,
        logger=logger,
    )
    atomic_input = filing_root / "06_atomic_input.csv"
    _run(
        [
            python, str(source / "06_atomize_core_mechanism_phrases.py"),
            "--input", str(decomposed), "--output", str(atomic_input),
            "--prompt", str(prompts / "06_atomize_core_mechanism_phrase_v5_7.txt"),
            "--cache-dir", str(filing_root / "06_core_atomization_cache"),
            "--model", model, "--workers", workers,
            "--batch-size", str(settings.get("atomize_batch_size", 4)),
            "--max-retries", str(settings.get("atomize_max_retries", 2)),
            "--max-tokens", str(settings.get("atomize_max_tokens", 4000)),
            "--request-timeout", str(settings.get("atomize_request_timeout", 90)),
            "--bisect-on-failure", "--fallback-to-parent-on-failure",
        ],
        env=env,
        logger=logger,
    )
    stage07 = filing_root / "07_atomic_audit"
    _run(
        [
            python, str(source / "07c_audit_atomic_contextual_v5_2.py"),
            "--input-csv", str(atomic_input), "--output-dir", str(stage07),
            "--prompt-file", str(prompts / "07_atomic_contextual_audit_v5_2.txt"),
            "--model", audit_model, "--api-base-url", env["DEEPSEEK_BASE_URL"],
            "--api-key-env", "DEEPSEEK_API_KEY", "--api-workers", workers,
            "--batch-size", "8",
            "--max-retries", str(settings.get("audit_max_retries", 3)),
            "--request-timeout", str(settings.get("audit_request_timeout", 90)),
            "--max-tokens", str(settings.get("audit_max_tokens", 4000)),
            "--bisect-on-failure", "--fallback-to-reject-on-failure",
        ],
        env=env,
        logger=logger,
    )
    atomic_keep = stage07 / "audited_keep.csv"
    if _count_rows(atomic_keep) == 0:
        return _finish_zero(filing_root, filing, "all_atomic_phrases_rejected", model, audit_model)

    raw_dedup = filing_root / "08_raw_dedup"
    dedup_base = [
        python, str(source / "11_dedup_mechanism_phrases.py"),
        "--semantic-prompt", str(prompts / "11_mechanism_semantic_dedup.md"),
        "--model", model, "--base-url", env["DEEPSEEK_BASE_URL"],
        "--api-key-env", "DEEPSEEK_API_KEY", "--api-workers", workers,
        "--pair-batch-size", "1", "--within-firm-year-only",
        "--preserve-material-qualifiers",
        "--max-retries", str(settings.get("dedup_max_retries", 2)),
        "--request-timeout", str(settings.get("dedup_request_timeout", 90)),
        "--max-tokens", str(settings.get("dedup_max_tokens", 2000)),
        "--conservative-no-merge-on-failure",
    ]
    _run(dedup_base + ["--input", str(atomic_keep), "--output-dir", str(raw_dedup), "--cache-dir", str(raw_dedup / "semantic_cache")], env=env, logger=logger)

    entity_dir = filing_root / "09_entity_role_audit"
    replaced = filing_root / "10_entity_replaced.csv"
    company = filing.get("company_name_wrds", "") or filing.get("ticker", "") or filing["sample_id"]
    _run(
        [
            python, str(source / "12_contextual_entity_replace.py"),
            "--input", str(raw_dedup / "mechanism_deduped.csv"), "--output", str(replaced),
            "--map-output", str(entity_dir / "generated_entity_replacement_map.csv"),
            "--firm", company, "--prompt", str(prompts / "12_contextual_entity_role_replacement_v5_7.txt"),
            "--cache-dir", str(entity_dir / "response_cache"), "--model", model,
            "--workers", str(min(8, int(workers))), "--batch-size", "8",
            "--max-retries", str(settings.get("entity_max_retries", 2)),
            "--request-timeout", str(settings.get("entity_request_timeout", 90)),
            "--max-tokens", str(settings.get("entity_max_tokens", 4000)),
            "--bisect-on-failure", "--fallback-to-no-replacements-on-failure",
        ],
        env=env,
        logger=logger,
    )
    residual = filing_root / "10b_entity_replaced_residual.csv"
    _run(
        [
            python, str(source / "12_contextual_entity_replace.py"),
            "--input", str(replaced), "--output", str(residual),
            "--map-output", str(entity_dir / "generated_residual_entity_replacement_map.csv"),
            "--firm", company, "--prompt", str(prompts / "12b_residual_entity_role_replacement_v5_7.txt"),
            "--cache-dir", str(entity_dir / "residual_response_cache"),
            "--phrase-field", "final_mechanism_phrase", "--candidate-hints",
            "--model", model, "--workers", str(min(8, int(workers))),
            "--batch-size", "8",
            "--max-retries", str(settings.get("entity_max_retries", 2)),
            "--request-timeout", str(settings.get("entity_request_timeout", 90)),
            "--max-tokens", str(settings.get("entity_max_tokens", 4000)),
            "--bisect-on-failure", "--fallback-to-no-replacements-on-failure",
        ],
        env=env,
        logger=logger,
    )
    final_dedup = filing_root / "11_final_dedup"
    _run(dedup_base + ["--input", str(residual), "--output-dir", str(final_dedup), "--cache-dir", str(final_dedup / "semantic_cache")], env=env, logger=logger)
    specificity_dir = filing_root / "12_specificity_audit"
    specificity_kept = specificity_dir / "specificity_kept.csv"
    specificity_audit = specificity_dir / "phrase_specificity_audit.csv"
    specificity_summary = specificity_dir / "specificity_summary.json"
    _run(
        [
            python, str(source / "13_audit_phrase_specificity.py"),
            "--input", str(final_dedup / "mechanism_deduped.csv"),
            "--kept-output", str(specificity_kept),
            "--audit-output", str(specificity_audit),
            "--summary-output", str(specificity_summary),
        ],
        env=env,
        logger=logger,
    )
    final_dir = filing_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "atom_mechanism_phrases_03_2.csv"
    _run(
        [
            python, str(source / "14_export_atom_phrases_slim.py"),
            "--input", str(specificity_kept), "--output", str(final_path),
        ],
        env=env,
        logger=logger,
    )
    summary = {
        "status": "pass",
        "pipeline_version": PHRASE_PIPELINE_VERSION,
        "extraction_model": model,
        "audit_model": audit_model,
        "sample_id": filing["sample_id"],
        "year": year,
        "accession": filing["accession_number"],
        "item1_sha256": filing.get("item1_sha256", ""),
        "stage01_rows": _count_rows(stage01 / "extraction_audit.csv"),
        "focal_candidates": _count_rows(curated),
        "candidate_audit_keep": _count_rows(audited),
        "atomic_audit_keep": _count_rows(atomic_keep),
        "specificity_audit_input": _count_rows(final_dedup / "mechanism_deduped.csv"),
        "specificity_audit_rejected": _count_rows(final_dedup / "mechanism_deduped.csv") - _count_rows(specificity_kept),
        "specificity_audit_output": str(specificity_audit.resolve()),
        "final_rows": _count_rows(final_path),
        "final_output": str(final_path.resolve()),
    }
    if repair_start > 1:
        summary["repair_start_stage"] = repair_start
    atomic_write_json(filing_root / "latest_pipeline_summary.json", summary)
    return summary


def _aggregate_latest_outputs(
    *,
    all_filings: list[dict[str, str]],
    output_root: Path,
    existing_summary: Callable[[Mapping[str, str]], dict[str, Any] | None],
    failed_filings: list[dict[str, str]],
    processed_this_run: int,
    offset: int,
    partial_selection: bool,
) -> dict[str, Any]:
    """Rebuild shared aggregates under an inter-process lock.

    Multiple Slurm array elements may finish at nearly the same time. Locking
    the complete rescan and atomic writes prevents a slower task from
    overwriting a newer, more complete aggregate with a stale snapshot.
    """

    lock_path = output_root / ".aggregate.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        completed_summaries = [
            summary
            for filing in all_filings
            if (summary := existing_summary(filing)) is not None
        ]
        final_rows: list[dict[str, str]] = []
        specificity_audit_rows: list[dict[str, str]] = []
        for summary in completed_summaries:
            path = Path(summary["final_output"])
            if path.is_file():
                final_rows.extend(read_csv(path))
            audit_path = Path(summary["specificity_audit_output"])
            if audit_path.is_file():
                for row in read_csv(audit_path):
                    specificity_audit_rows.append(
                        {
                            "sample_id": summary["sample_id"],
                            "accession": summary["accession"],
                            **row,
                        }
                    )
        aggregate = output_root / "all_final_mechanism_phrases.csv"
        fields = list(final_rows[0]) if final_rows else FINAL_FIELDS
        atomic_write_csv(aggregate, final_rows, fields)
        specificity_aggregate = output_root / "all_phrase_specificity_audit.csv"
        audit_fields = list(specificity_audit_rows[0]) if specificity_audit_rows else [
            "sample_id",
            "accession",
            *SPECIFICITY_AUDIT_FIELDS,
        ]
        atomic_write_csv(specificity_aggregate, specificity_audit_rows, audit_fields)
        missing_filings = [
            {
                "sample_id": filing["sample_id"],
                "year": filing["target_fiscal_year"],
                "accession": filing["accession_number"],
            }
            for filing in all_filings
            if existing_summary(filing) is None
        ]
        complete = not failed_filings and not missing_filings
        # Slurm array tasks operate on partial selections. A filing-level
        # failure remains visible without blocking other array elements.
        status = "pass" if complete else ("partial" if partial_selection else "fail")
        result = {
            "status": status,
            "pipeline_version": PHRASE_PIPELINE_VERSION,
            "pipeline": "historical_v5.8_with_cross_industry_generic_prompts",
            "generated_at_utc": utc_now(),
            "filings": len(completed_summaries),
            "target_filings": len(all_filings),
            "processed_this_run": processed_this_run,
            "offset": offset,
            "final_rows": len(final_rows),
            "aggregate_output": str(aggregate.resolve()),
            "specificity_audit_rows": len(specificity_audit_rows),
            "specificity_audit_rejected": sum(
                int(summary.get("specificity_audit_rejected", 0))
                for summary in completed_summaries
            ),
            "specificity_audit_output": str(specificity_aggregate.resolve()),
            "filing_summaries": completed_summaries,
            "failed_filings": failed_filings,
            "missing_filings": missing_filings,
        }
        atomic_write_json(output_root / "latest_pipeline_summary.json", result)
        return result


def extract_latest_phrases(
    config: Mapping[str, Any], project_root: Path, logger: logging.Logger,
    *, limit: int | None = None, offset: int = 0, sample_id: str | None = None,
) -> dict[str, Any]:
    settings = config.get("latest_phrase_pipeline")
    if not isinstance(settings, Mapping):
        raise ValueError("latest_phrase_pipeline configuration is required")
    env = _api_environment(settings, project_root)
    manifest_path = resolve_project_path(project_root, settings.get("clean_manifest", "outputs/manifests/clean_manifest.csv"))
    all_filings = [row for row in read_csv(manifest_path) if row.get("clean_status") == "pass"]
    filings = all_filings
    if sample_id:
        filings = [row for row in filings if row.get("sample_id") == sample_id]
    if offset < 0:
        raise ValueError("phrase offset must be non-negative")
    if offset:
        filings = filings[offset:]
    if limit is not None:
        filings = filings[:limit]
    if not filings:
        raise ValueError("no matching clean filings for the latest phrase pipeline")
    summaries = []
    failed_filings: list[dict[str, str]] = []
    output_root = resolve_project_path(project_root, settings.get("output_root", "outputs/phrases_latest"))
    output_root.mkdir(parents=True, exist_ok=True)

    def existing_summary(filing: Mapping[str, str]) -> dict[str, Any] | None:
        summary_path = (
            output_root
            / filing["sample_id"]
            / filing["target_fiscal_year"]
            / filing["accession_number"]
            / "latest_pipeline_summary.json"
        )
        if not summary_path.is_file():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final_path = Path(str(summary.get("final_output", "")))
            expected_rows = int(summary.get("final_rows", -1))
            audit_path = Path(str(summary.get("specificity_audit_output", "")))
            audit_input = int(summary.get("specificity_audit_input", -1))
            audit_rejected = int(summary.get("specificity_audit_rejected", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if (
            summary.get("status") != "pass"
            or summary.get("pipeline_version") != PHRASE_PIPELINE_VERSION
            or summary.get("sample_id") != filing["sample_id"]
            or str(summary.get("year")) != filing["target_fiscal_year"]
            or summary.get("accession") != filing["accession_number"]
            or summary.get("item1_sha256") != filing.get("item1_sha256", "")
            or (summary.get("audit_model") is not None and summary.get("audit_model") != str(settings.get("audit_model", settings.get("model", "openai/gpt-oss-20b"))))
            or not final_path.is_file()
            or _count_rows(final_path) != expected_rows
            or not audit_path.is_file()
            or _count_rows(audit_path) != audit_input
            or expected_rows + audit_rejected != audit_input
        ):
            return None
        return summary

    resume = bool(config.get("pipeline", {}).get("resume", True))
    for number, filing in enumerate(filings, 1):
        logger.info("Latest phrase filing %d/%d: %s", number, len(filings), filing["sample_id"])
        cached_summary = existing_summary(filing) if resume else None
        if cached_summary is not None:
            logger.info(
                "Skipping completed filing %s %s %s (%s final rows)",
                filing["sample_id"], filing["target_fiscal_year"],
                filing["accession_number"], cached_summary["final_rows"],
            )
            summaries.append(cached_summary)
        else:
            try:
                summaries.append(_run_filing(filing, settings, project_root, env, logger))
            except Exception as error:
                # Isolate a filing-level failure so one unusual SEC document
                # cannot prevent the rest of a Slurm chunk from being tried.
                # run.sbatch will still retry the chunk; completed filing
                # summaries are then skipped and only failures are retried.
                logger.exception(
                    "Latest phrase filing failed but the chunk will continue: %s %s %s",
                    filing["sample_id"], filing["target_fiscal_year"],
                    filing["accession_number"],
                )
                failed_filings.append(
                    {
                        "sample_id": filing["sample_id"],
                        "year": filing["target_fiscal_year"],
                        "accession": filing["accession_number"],
                        "error": repr(error),
                    }
                )

    partial_selection = limit is not None or offset != 0 or sample_id is not None
    return _aggregate_latest_outputs(
        all_filings=all_filings,
        output_root=output_root,
        existing_summary=existing_summary,
        failed_filings=failed_filings,
        processed_this_run=len(filings),
        offset=offset,
        partial_selection=partial_selection,
    )
