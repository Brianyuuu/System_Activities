"""Shared validation, hashing, atomic output, and logging utilities."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("pipeline configuration must be a JSON object")
    return value


def validate_config(config: Mapping[str, Any], project_root: Path) -> None:
    for section in ("wrds", "pipeline", "cleaning"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} is required")
    pipeline = config["pipeline"]
    for key in ("targets", "output_root", "allowed_forms"):
        if not pipeline.get(key):
            raise ValueError(f"pipeline.{key} is required")
    target_path = resolve_project_path(project_root, pipeline["targets"])
    if not target_path.is_file():
        raise FileNotFoundError(f"target CSV does not exist: {target_path}")
    if pipeline.get("identity_bridge"):
        identity_path = resolve_project_path(project_root, pipeline["identity_bridge"])
        if not identity_path.is_file():
            raise FileNotFoundError(f"identity bridge does not exist: {identity_path}")
    if int(config["cleaning"].get("min_item1_words", 0)) < 1:
        raise ValueError("cleaning.min_item1_words must be positive")
    phrase = config.get("phrase_extraction")
    if phrase is not None:
        if not isinstance(phrase, dict):
            raise ValueError("phrase_extraction must be a JSON object")
        for key in ("prompt_file", "repair_prompt_file", "output_root"):
            if not phrase.get(key):
                raise ValueError(f"phrase_extraction.{key} is required")
        for key in ("prompt_file", "repair_prompt_file"):
            path = resolve_project_path(project_root, phrase[key])
            if not path.is_file():
                raise FileNotFoundError(f"phrase prompt does not exist: {path}")
    latest = config.get("latest_phrase_pipeline")
    if latest is not None:
        if not isinstance(latest, dict):
            raise ValueError("latest_phrase_pipeline must be a JSON object")
        for key in ("source_root", "prompt_root", "output_root"):
            if not latest.get(key):
                raise ValueError(f"latest_phrase_pipeline.{key} is required")
        source_root = resolve_project_path(project_root, latest["source_root"])
        prompt_root = resolve_project_path(project_root, latest["prompt_root"])
        required_sources = (
            "01b_context_clausal_actor_extraction_v5_5.py",
            "03_2_audit_candidate_mechanism_type_v5_3.py",
            "04_prepare_unmodified_mechanism_input.py",
            "05_decompose_mechanism_phrases.py",
            "06_atomize_core_mechanism_phrases.py",
            "07c_audit_atomic_contextual_v5_2.py",
            "11_dedup_mechanism_phrases.py",
            "12_contextual_entity_replace.py",
            "13_audit_phrase_specificity.py",
            "14_export_atom_phrases_slim.py",
            "item1_boundary_v5_5.py",
        )
        required_prompts = (
            "01_actor_context_clausal_v5_3.txt",
            "01_actor_context_clausal_repair_v5_3.txt",
            "03_2_candidate_mechanism_type_v5_3.txt",
            "05_decompose_mechanism_phrases_generic.txt",
            "06_atomize_core_mechanism_phrase_v5_7.txt",
            "07_atomic_contextual_audit_v5_2.txt",
            "11_mechanism_semantic_dedup.md",
            "12_contextual_entity_role_replacement_v5_7.txt",
            "12b_residual_entity_role_replacement_v5_7.txt",
        )
        for path in [*(source_root / name for name in required_sources), *(prompt_root / name for name in required_prompts)]:
            if not path.is_file():
                raise FileNotFoundError(f"latest phrase pipeline asset does not exist: {path}")


def configure_logging(log_dir: Path, level: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{stamp}.log"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys_stdout()), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return logging.getLogger("wrds_10k_pipeline"), log_path


def sys_stdout():
    import sys

    return sys.stdout


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(str(key))
                    seen.add(str(key))
        fieldnames = ordered
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{str(k): clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def clean(value: Any) -> str:
    if value is None:
        return ""
    value_type = type(value)
    if value_type.__name__ in {"NAType", "NaTType"} and value_type.__module__.startswith(
        ("pandas", "numpy")
    ):
        return ""
    try:
        if value != value:  # NaN without importing pandas in this utility module.
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_cik(value: Any) -> str:
    digits = re.sub(r"\D", "", clean(value))
    if not digits or len(digits) > 10:
        raise ValueError(f"invalid CIK: {value!r}")
    return digits.zfill(10)


def normalize_accession(value: Any) -> str:
    text = clean(value)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    if len(digits) != 18:
        raise ValueError(f"invalid accession: {value!r}")
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def safe_component(value: Any, label: str) -> str:
    text = clean(value)
    if not text or Path(text).name != text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe {label}: {value!r}")
    return text


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_date(value: Any) -> date | None:
    text = clean(value)[:10]
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def fiscal_year(report_date: Any, filing_date: Any) -> int | None:
    report = parse_date(report_date)
    if report:
        return report.year - 1 if report.month == 1 and report.day <= 7 else report.year
    filed = parse_date(filing_date)
    return filed.year if filed else None


def validate_targets(
    rows: Iterable[Mapping[str, Any]],
    identity_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Normalize either the canonical schema or the Russell sample schema.

    Canonical columns are sample_id/cik/target_fiscal_year.  The supplied
    random sample uses entity_id/permno/year. A frozen bridge may supply an
    initial CIK, but a PERMNO is sufficient because the planning stage resolves
    dated CIK links from WRDS. No ticker-only matching is allowed.
    """
    bridge_by_entity: dict[str, str] = {}
    bridge_by_permno: dict[str, str] = {}
    for bridge in identity_rows or ():
        entity = clean(bridge.get("entity_id") or bridge.get("sample_id"))
        permno = clean(bridge.get("permno") or bridge.get("PERMNO"))
        cik_value = clean(bridge.get("cik"))
        if not cik_value:
            continue
        cik = normalize_cik(cik_value)
        if entity:
            if entity in bridge_by_entity and bridge_by_entity[entity] != cik:
                raise ValueError(f"identity bridge has conflicting CIKs for {entity}")
            bridge_by_entity[entity] = cik
        if permno:
            if permno in bridge_by_permno and bridge_by_permno[permno] != cik:
                raise ValueError(f"identity bridge has conflicting CIKs for PERMNO {permno}")
            bridge_by_permno[permno] = cik

    output: list[dict[str, str]] = []
    keys: set[tuple[str, int]] = set()
    for index, source in enumerate(rows, 2):
        row = {str(key): clean(value) for key, value in source.items()}
        row["sample_id"] = clean(row.get("sample_id") or row.get("entity_id"))
        row["target_fiscal_year"] = clean(row.get("target_fiscal_year") or row.get("year"))
        if not row["sample_id"] or not row.get("ticker") or not row["target_fiscal_year"]:
            raise ValueError(
                f"target CSV line {index} requires sample_id/entity_id, ticker, "
                "and target_fiscal_year/year"
            )
        cik_value = clean(row.get("cik"))
        if not cik_value:
            cik_value = bridge_by_entity.get(row["sample_id"], "")
        if not cik_value and row.get("permno"):
            cik_value = bridge_by_permno.get(row["permno"], "")
        if not cik_value and not row.get("permno"):
            raise ValueError(
                f"target CSV line {index} requires either CIK or PERMNO: "
                f"{row['sample_id']}"
            )
        row["cik"] = normalize_cik(cik_value) if cik_value else ""
        year = int(row["target_fiscal_year"])
        if year < 1994 or year > datetime.now().year + 1:
            raise ValueError(f"invalid target year on CSV line {index}: {year}")
        row["target_fiscal_year"] = str(year)
        row["accession_number"] = normalize_accession(row.get("accession_number", ""))
        key = (row["sample_id"], year)
        if key in keys:
            raise ValueError(f"duplicate target: {key}")
        keys.add(key)
        output.append(row)
    if not output:
        raise ValueError("target CSV is empty")
    return sorted(output, key=lambda row: (row["sample_id"], int(row["target_fiscal_year"])))
