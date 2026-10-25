"""Extract the primary 10-K document and a fail-closed Item 1 text span."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from bs4 import BeautifulSoup

from utils import atomic_write_csv, atomic_write_text, read_csv, resolve_project_path, sha256_file


ITEM1_EXTRACTOR_VERSION = "unified-primary-v2-2026-09-14"
DOCUMENT_RE = re.compile(br"(?is)<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)")
TYPE_RE = re.compile(br"(?im)^<TYPE>\s*([^\r\n<]+)")
TEXT_RE = re.compile(br"(?is)<TEXT>(.*?)(?:</TEXT>|\Z)")
START_RE = re.compile(
    r"(?im)^[ \t]*(?:part[ \t]+i[ \t]*)?(?:"
    r"(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*s?[ \t\r\n]*)?"
    r"(?:1|i)[ \t\r\n]*(?:\.*[ \t\r\n]*(?:and|&)[ \t\r\n]*2[ \t]*)?"
    r"[.\-|:\u2013\u2014\s]*"
    r"(?:description[ \t\r\n]+of[ \t\r\n]+)?"
    r"(?:our[ \t\r\n]+)?"
    r"b[ \t\r\n]*u[ \t\r\n]*s[ \t\r\n]*i[ \t\r\n]*n[ \t\r\n]*e[ \t\r\n]*s[ \t\r\n]*s"
    r"(?:[ \t]+(?:of[ \t]+[^\r\n]{1,90}|overview))?"
    r"(?:[ \t\r\n]+and[ \t\r\n]+propert(?:y|ies))?"
    r"|(?:description[ \t\r\n]+of[ \t\r\n]+)?"
    r"(?:our[ \t\r\n]+)?"
    r"b[ \t\r\n]*u[ \t\r\n]*s[ \t\r\n]*i[ \t\r\n]*n[ \t\r\n]*e[ \t\r\n]*s[ \t\r\n]*s"
    r"(?:[ \t]+overview)?"
    r")[ \t.:-]*(?:\([^)]{1,160}\))?[ *\t.:-]*$"
)
START_ALT_RE = re.compile(r"(?im)^[ \t]*item[ \t]*\.[ \t]*1[ \t.:-]*business[ \t.:-]*$")

# A valid SEC heading can be split over several HTML nodes, so the boundary
# regexes intentionally accept embedded newlines.  That also means a linked
# cross-reference such as ``see <a>Item 1A. Risk Factors</a>`` can look like a
# standalone heading after HTML-to-text conversion.  Reject those candidates
# using the prose immediately surrounding the match before pairing boundaries.
REFERENCE_BEFORE_RE = re.compile(
    r"(?i)(?:"
    r"see(?:\s+(?:also|further))?(?:\s+(?:discussion|information))?(?:\s+in)?|"
    r"refer(?:red|ring)?\s+to|"
    r"described\s+in|discussed\s+in|set\s+forth\s+in|included\s+in|"
    r"contained\s+in|identified\s+in|outlined\s+in|addressed\s+in|"
    r"presented\s+in|summarized\s+in|found\s+in|under|pursuant\s+to"
    r")\s*[“\"']?\s*(?:item\s*)?$"
)
REFERENCE_AFTER_RE = re.compile(
    r"(?i)^\s*(?:[.,;:]?\s*[”\"']\s*)?(?:"
    r"\)|,|;|which\s+is\s+incorporated|for\s+(?:further|additional|more|a\s+discussion|information)|"
    r"of\s+this\s+(?:annual\s+report|form\s+10-k)|above|below"
    r")"
)


def _is_cross_reference(text: str, match: re.Match[str]) -> bool:
    """Return True when an apparent Item heading is an inline citation."""

    before = text[max(0, match.start() - 260) : match.start()]
    after = text[match.end() : match.end() + 180]
    # HTML links frequently put the opening quote or the word "See" on its
    # own line.  Whitespace normalization lets the same rule cover both HTML
    # and legacy plain-text filings.
    before_flat = re.sub(r"\s+", " ", before).rstrip()
    after_flat = re.sub(r"\s+", " ", after)
    # Formal SEC headings are not enclosed in quotation marks.  A matched
    # title ending in a closing quote is therefore an unambiguous prose
    # reference even when the introducing "see" is separated by a long title.
    if re.search(r"[”\"']\s*$", match.group(0).rstrip()):
        return True
    if REFERENCE_BEFORE_RE.search(before_flat[-180:]):
        return True
    context_tail = before_flat[-260:]
    cue_matches = list(re.finditer(r"(?i)\b(?:see|refer(?:red|ring)?)\b", context_tail))
    structural_marker = max(
        context_tail.lower().rfind("table of contents"),
        context_tail.lower().rfind("part i"),
        context_tail.lower().rfind("part 1"),
    )
    if cue_matches and cue_matches[-1].start() > structural_marker:
        # Only treat the cue as introducing this match when no sentence ended
        # between them.  SEC boilerplate often ends with "incorporated by
        # reference." followed by a page number and the real Item heading.
        after_cue = context_tail[cue_matches[-1].end() :]
        if not re.search(r"[.!?][”\"')\]]*", after_cue):
            return True
    if REFERENCE_AFTER_RE.search(after_flat):
        # A closing quote/parenthesis followed by prose is a citation.  Do not
        # reject a real heading merely because its first sentence starts with
        # the word "For".
        matched = match.group(0).rstrip()
        return bool(
            re.search(r"[”\"']\s*$", matched)
            or re.match(r"^\s*[”\"')],", after)
            or re.match(r"^\s*[”\"']?\s*(?:\)|,|;|of\s+this|above|below)", after_flat, re.I)
        )
    return False
END_RE = re.compile(
    r"(?im)^[ \t]*(?:part[ \t]+i[ \t]*)?(?:"
    r"(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)"
    r"1[ \t\r\n]*(?:a|\([ \t]*a[ \t]*\))(?![A-Za-z0-9])"
    r"[^\r\n]*(?:\r?\n[ \t]*risk[ \t]+factors)?"
    r"|1[ \t]*(?:a|\([ \t]*a[ \t]*\))(?![A-Za-z0-9])"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?risk[ \t]+factors[^\r\n]*"
    r"|(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)"
    r"1[ \t\r\n]*(?:b|\([ \t]*b[ \t]*\))(?![A-Za-z0-9])[^\r\n]*"
    r"|1[ \t]*(?:b|\([ \t]*b[ \t]*\))(?![A-Za-z0-9])"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?unresolved[ \t]+staff[ \t]+comments[^\r\n]*"
    r"|(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)?2[ \t\r\n]*"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?"
    r"(?:description[ \t]+of[ \t]+)?propert(?:y|ies)\b[^\r\n]*"
    r"|(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*3\b[^\r\n]*"
    r"|3[.\-:\u2013\u2014 \t]+legal[ \t]+proceedings\b[^\r\n]*"
    r"|risk[ \t]+factors"
    r"|properties"
    r")[ \t.]*$"
)
FORMAL_END_RE = re.compile(
    r"(?im)^[ \t]*(?:part[ \t]+i[ \t]*)?(?:"
    r"(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)"
    r"1[ \t\r\n]*(?:a|\([ \t]*a[ \t]*\))(?![A-Za-z0-9])"
    r"[^\r\n]*(?:\r?\n[ \t]*risk[ \t]+factors)?"
    r"|1[ \t]*(?:a|\([ \t]*a[ \t]*\))(?![A-Za-z0-9])"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?risk[ \t]+factors[^\r\n]*"
    r"|(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)"
    r"1[ \t\r\n]*(?:b|\([ \t]*b[ \t]*\))(?![A-Za-z0-9])[^\r\n]*"
    r"|1[ \t]*(?:b|\([ \t]*b[ \t]*\))(?![A-Za-z0-9])"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?unresolved[ \t]+staff[ \t]+comments[^\r\n]*"
    r"|(?:(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*)?2[ \t\r\n]*"
    r"[.\-:\u2013\u2014 \t]*(?:\r?\n[ \t]*)?"
    r"(?:description[ \t]+of[ \t]+)?propert(?:y|ies)\b[^\r\n]*"
    r"|(?:i[ \t\r\n]*)?t[ \t\r\n]*e[ \t\r\n]*m[ \t\r\n]*3\b[^\r\n]*"
    r"|3[.\-:\u2013\u2014 \t]+legal[ \t]+proceedings\b[^\r\n]*"
    r")[ \t.]*$"
)


def _primary_document(payload: bytes, allowed_forms: set[str]) -> bytes:
    documents = []
    for match in DOCUMENT_RE.finditer(payload):
        block = match.group(1)
        form_match = TYPE_RE.search(block[:50000])
        form = form_match.group(1).decode("latin-1", "replace").strip().upper() if form_match else ""
        text_match = TEXT_RE.search(block)
        body = text_match.group(1) if text_match else block
        if form in allowed_forms:
            documents.append(body.strip())
    if documents:
        # The SEC complete-submission convention places the primary filing
        # document first. Some older submissions repeat TYPE=10-K on an
        # exhibit or printable copy, so requiring exactly one loses valid
        # filings such as CAT and CCL.
        return documents[0]
    return payload


def _visible_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", "replace")
    if decoded.count("\ufffd") > max(10, len(decoded) // 1000):
        decoded = payload.decode("latin-1", "replace")
    soup = BeautifulSoup(decoded, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_item1(text: str, min_words: int) -> tuple[str, int, int]:
    starts = [
        match for pattern in (START_RE, START_ALT_RE)
        for match in pattern.finditer(text) if not _is_cross_reference(text, match)
    ]
    starts.sort(key=lambda match: match.start())
    ends = [match for match in END_RE.finditer(text) if not _is_cross_reference(text, match)]
    formal_ends = [
        match for match in FORMAL_END_RE.finditer(text) if not _is_cross_reference(text, match)
    ]
    eligible: list[tuple[int, int, int, int, str]] = []
    for start in starts:
        # Always pair a heading with its *immediate* following section
        # boundary. A table of contents commonly contains a valid-looking
        # "Item 1. Business" line followed almost immediately by Item 1A.
        # Skipping that short boundary in search of a later, long span makes
        # the TOC look like the start of the real section and contaminates the
        # result with most of the filing.
        # Prefer numbered Item boundaries. Bare "Risk Factors" and
        # "Properties" headings occur inside legitimate business narratives,
        # so they are only safe as a fallback for legacy filings that have no
        # later numbered boundary at all.
        end = next(
            (candidate for candidate in formal_ends if candidate.start() > start.end()),
            None,
        )
        if end is None:
            end = next((candidate for candidate in ends if candidate.start() > start.end()), None)
        if end is None:
            continue
        section = text[start.end() : end.start()].strip()
        words = len(re.findall(r"\b[\w'-]+\b", section))
        # A boundary that runs through most of a filing is usually a table of
        # contents heading paired with a much later section. Fail closed.
        if min_words <= words <= 50000:
            explicit_item = int(
                bool(
                    re.search(
                        r"(?i)\b(?:i\s*)?t\s*e\s*m\s*s?\s*(?:1|i)\b",
                        start.group(0),
                    )
                )
            )
            eligible.append((explicit_item, words, start.start(), end.start(), section))
    if not eligible:
        raise ValueError(
            f"no Item 1 span met the {min_words}-word minimum; starts={len(starts)} ends={len(ends)}"
        )
    _explicit_item, words, start_char, end_char, section = max(
        eligible, key=lambda value: (value[0], value[1])
    )
    return section, start_char, end_char


def clean_downloads(
    config: Mapping[str, Any],
    project_root: Path,
    download_manifest: Path,
    logger: logging.Logger,
) -> Path:
    pipeline = config["pipeline"]
    output_root = resolve_project_path(project_root, pipeline["output_root"])
    clean_manifest = output_root / "manifests" / "clean_manifest.csv"
    resume = bool(pipeline.get("resume", True))
    min_words = int(config["cleaning"]["min_item1_words"])
    allowed_forms = {str(value).upper() for value in pipeline["allowed_forms"]}
    rows: list[dict[str, Any]] = []

    for position, source in enumerate(read_csv(download_manifest), 1):
        if position == 1 or position % 25 == 0:
            logger.info("Canonical Item 1 progress: %d filings processed", position - 1)
        result: dict[str, Any] = {**source, "clean_status": "error", "clean_error": ""}
        if source.get("status") != "pass":
            result["clean_error"] = "download did not pass"
            rows.append(result)
            continue
        try:
            raw_path = Path(source["raw_path"])
            item1_path = (
                output_root
                / "item1"
                / source["sample_id"]
                / source["target_fiscal_year"]
                / f"{source['accession_number']}_item1.md"
            )
            use_cached = resume and item1_path.is_file() and item1_path.stat().st_size > 0
            marker = "\n# Item 1. Business\n\n"
            if use_cached:
                markdown = item1_path.read_text(encoding="utf-8")
                if (
                    marker not in markdown
                    or f'"item1_extractor": "{ITEM1_EXTRACTOR_VERSION}"' not in markdown
                ):
                    # An older extractor may have silently stopped at an inline
                    # Item citation.  Rebuild that cache once instead of
                    # allowing resume mode to preserve a stale boundary.
                    use_cached = False
            if use_cached:
                section = markdown.split(marker, 1)[1]
                words = len(re.findall(r"\b[\w'-]+\b", section))
                if words < min_words:
                    use_cached = False
            if use_cached:
                start_char = end_char = "cached"
            else:
                payload = raw_path.read_bytes()
                primary = _primary_document(payload, allowed_forms)
                visible = _visible_text(primary)
                section, start_char, end_char = _extract_item1(visible, min_words)
                words = len(re.findall(r"\b[\w'-]+\b", section))
                metadata = {
                    "sample_id": source["sample_id"],
                    "ticker": source["ticker"],
                    "cik": source["cik"],
                    "target_fiscal_year": int(source["target_fiscal_year"]),
                    "accession_number": source["accession_number"],
                    "form": source["form"],
                    "raw_sha256": source["raw_sha256"],
                    "source_method": source["source_method"],
                    "item1_extractor": ITEM1_EXTRACTOR_VERSION,
                }
                markdown = "---\nmetadata: " + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                markdown += "\n---\n\n# Item 1. Business\n\n" + section.strip() + "\n"
                atomic_write_text(item1_path, markdown)
            result.update(
                {
                    "item1_path": str(item1_path.resolve()),
                    "item1_sha256": sha256_file(item1_path),
                    "item1_words": words,
                    "boundary_start_char": start_char,
                    "boundary_end_char": end_char,
                    "item1_extractor": ITEM1_EXTRACTOR_VERSION,
                    "clean_status": "pass",
                }
            )
        except Exception as exc:
            result["clean_error"] = f"{type(exc).__name__}: {exc}"
            logger.error("[%d] %s", position, result["clean_error"])
        rows.append(result)

    logger.info("Canonical Item 1 progress: %d filings processed", len(rows))
    atomic_write_csv(clean_manifest, rows)
    failures = [row for row in rows if row["clean_status"] != "pass"]
    allow_partial = bool(pipeline.get("allow_partial_item1", False))
    if failures and bool(pipeline.get("strict", True)) and not allow_partial:
        raise RuntimeError(f"Item 1 cleaning failed for {len(failures)}/{len(rows)} targets; see {clean_manifest}")
    if failures and allow_partial:
        logger.warning(
            "Queued %d/%d Item 1 failures for repair; continuing with %d valid filings",
            len(failures),
            len(rows),
            len(rows) - len(failures),
        )
    return clean_manifest
