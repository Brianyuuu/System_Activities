#!/usr/bin/env python3
"""Robust, auditable SEC 10-K Item 1 boundary detection.

This module deliberately separates local document-boundary detection from LLM
extraction.  It rejects table-of-contents markers and inline cross-references,
then pairs each credible Item 1 heading with the first credible following
Item 1A/Item 2 heading.  Diagnostics are returned for every filing so silent
truncation cannot pass as a successful run.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from lxml import html as lxml_html


SPACE = re.compile(r"\s+")
ITEM1_START = re.compile(
    # Reject Item 1A/1B/1C and multi-digit items, but do not reject an
    # unpunctuated heading such as "ITEM 1 BUSINESS" merely because Business
    # begins with B.  The former `(?!\s*[a-c0-9])` made that common SEC format
    # impossible to match.
    r"\bi\s*t\s*e\s*m\s+(?:"
    r"1(?!\s*[a-c](?=\s|[.\-–—:]))(?!\s*[0-9])|"
    r"i(?!\s*[a-z0-9])"
    r")[\s.\-–—:]*"
    r"(?:[*_`\\|]+\s*)*(?:description\s+of\s+)?(?:our\s+)?"
    r"b\s*u\s*s\s*i\s*n\s*e\s*s\s*s\b",
    re.I,
)
ITEMS_1_2_START = re.compile(
    r"\bi\s*t\s*e\s*m\s*s\s+1[\s.]*"
    r"(?:&|a\s*n\s*d)\s*2[\s.\-–—:]*"
    r"b\s*u\s*s\s*i\s*n\s*e\s*s\s*s\s+"
    r"(?:&|a\s*n\s*d)\s+p\s*r\s*o\s*p\s*e\s*r\s*t\s*i\s*e\s*s\b",
    re.I,
)
ITEM1A_END = re.compile(
    r"\bi\s*t\s*e\s*m\s+1\s*a[\s.\-–—:]*"
    r"(?:[*_`\\|]+\s*)*r\s*i\s*s\s*k\s+f\s*a\s*c\s*t\s*o\s*r\s*s\b",
    re.I,
)
ITEM2_END = re.compile(
    r"\bi\s*t\s*e\s*m\s+2[\s.\-–—:]*"
    r"(?:[*_`\\|]+\s*)*(?:description\s+of\s+)?"
    r"p\s*r\s*o\s*p\s*e\s*r\s*t(?:y|i\s*e\s*s)\b",
    re.I,
)
ITEM3_END = re.compile(
    r"\bi\s*t\s*e\s*m\s+3[\s.\-–—:]*"
    r"(?:[*_`\\|]+\s*)*l\s*e\s*g\s*a\s*l\s+p\s*r\s*o\s*c\s*e\s*e\s*d\s*i\s*n\s*g\s*s\b",
    re.I,
)
PART_I_BUSINESS_START = re.compile(
    r"\bpart\s+i\b\s+(?:"
    r"i\s*t\s*e\s*m\s+1(?!\s*[a-c](?=\s|[.\-–—:]))(?!\s*[0-9])\s*[.\-–—:]?\s*"
    r"b\s*u\s*s\s*i\s*n\s*e\s*s\s*s\b|overview\b)",
    re.I,
)

START_CROSS_REFERENCE = re.compile(
    r"(?i)(?:see|refer(?:red|ring)?\s+to|described\s+in|discussed\s+in|under)\s*"
    r"(?:part\s+[ivx]+\s*[,.;:]?\s*)?$"
)
END_CROSS_REFERENCE_BEFORE = re.compile(
    r"(?i)(?:see\s+(?:further\s+)?(?:discussion|information)\s+in|see|also\s+see|"
    r"refer(?:red|ring)?\s+to|described\s+in|discussed\s+in|"
    r"set\s+forth\s+in|included\s+in|contained\s+in|identified\s+in|outlined\s+in|"
    r"addressed\s+in|presented\s+in|summarized\s+in|found\s+in|under)\s*"
    r"(?:part\s+(?:[ivx]+|\d+)\s*[,.;:]?\s*)?[“\"']?$"
)
END_CROSS_REFERENCE_AFTER = re.compile(
    r"(?i)^\s*(?:[.,;:]?\s*[”\"']|[-–—]\s*[“\"']|for\s+(?:discussion|a\s+description|additional\s+information)|"
    r"of\s+this\s+form\s+10-k)"
)


@dataclass(frozen=True)
class BoundaryResult:
    source_file: str
    visible_words: int
    section_words: int
    start_char: int
    end_char: int
    start_text: str
    end_text: str
    start_candidates: int
    end_candidates: int
    rejected_toc_starts: int
    rejected_cross_reference_starts: int
    rejected_cross_reference_ends: int
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_text(value: str) -> str:
    return SPACE.sub(" ", html.unescape(value).replace("\xa0", " ")).strip()


def html_to_text(path: Path) -> str:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8", "replace")
    if "<html" not in decoded[:20_000].lower() and "<body" not in decoded[:20_000].lower():
        return normalize_text(decoded)
    parser = lxml_html.HTMLParser(encoding="utf-8", recover=True)
    root = lxml_html.fromstring(raw, parser=parser)
    for node in root.xpath("//script|//style|//noscript|//svg"):
        node.drop_tree()
    return normalize_text(" ".join(root.itertext()))


def _letters_are_upper(value: str) -> bool:
    letters = "".join(char for char in value if char.isalpha())
    return bool(letters) and letters == letters.upper()


def _context(text: str, start: int, end: int, before: int = 100, after: int = 140) -> tuple[str, str]:
    return text[max(0, start - before):start], text[end:end + after]


def _is_toc_start(text: str, match: re.Match[str]) -> bool:
    nearby = text[match.end():match.end() + 650]
    end = ITEM1A_END.search(nearby) or ITEM2_END.search(nearby) or ITEM3_END.search(nearby)
    return bool(end and len(nearby[:end.start()].split()) < 120)


def _is_cross_reference_start(text: str, match: re.Match[str]) -> bool:
    before, after = _context(text, match.start(), match.end())
    if START_CROSS_REFERENCE.search(before):
        return True
    # A matched title immediately followed by a closing quote is a narrative
    # citation (for example, `under “Item 1. Business” as well as ...`), not a
    # section heading. This form otherwise caused a later Item 1A citation to
    # replace the real Item 1 start when the longest candidate was selected.
    if re.match(r"^\s*[.,;:]?\s*[”\"']", after):
        return True
    # Historical Markdown conversions often retain a dashed underline directly
    # after the real heading.  It is layout, not an inline subsection reference.
    if re.match(r"^\s*[-–—]{3,}\s+", after) or re.match(
        r"^\s*(?:[-–—]+\s+){2,}", after
    ):
        return False
    # Inline references normally continue with a dash/quoted subsection.  A
    # real heading may continue with Overview/General, so do not reject those.
    return bool(re.match(r"^\s*[-–—]\s*[“\"']?", after))


def _is_cross_reference_end(text: str, match: re.Match[str]) -> bool:
    before, after = _context(text, match.start(), match.end())
    return bool(
        END_CROSS_REFERENCE_BEFORE.search(before)
        or END_CROSS_REFERENCE_AFTER.search(after)
    )


def extract_item1(
    text: str,
    *,
    source_file: str = "",
    min_words: int = 2_000,
    max_words: int = 100_000,
    warning_words: int = 3_000,
) -> tuple[str, BoundaryResult]:
    regular_starts = list(ITEM1_START.finditer(text))
    combined_starts = list(ITEMS_1_2_START.finditer(text))
    starts_all = sorted([*regular_starts, *combined_starts], key=lambda match: match.start())
    combined_start_ids = {id(match) for match in combined_starts}
    item1a_ends = list(ITEM1A_END.finditer(text))
    item2_ends = list(ITEM2_END.finditer(text))
    item3_ends = list(ITEM3_END.finditer(text))
    standard_ends = sorted([*item1a_ends, *item2_ends], key=lambda match: match.start())
    combined_ends = sorted([*item1a_ends, *item3_ends], key=lambda match: match.start())
    ends_all = sorted(
        {match.start(): match for match in [*standard_ends, *combined_ends]}.values(),
        key=lambda match: match.start(),
    )
    if not starts_all or not ends_all:
        raise ValueError(
            f"Could not locate Item 1 boundaries for {source_file}: "
            f"starts={len(starts_all)} ends={len(ends_all)}"
        )

    toc_starts = {id(match) for match in starts_all if _is_toc_start(text, match)}
    cross_starts = {id(match) for match in starts_all if _is_cross_reference_start(text, match)}
    cross_ends = {id(match) for match in ends_all if _is_cross_reference_end(text, match)}

    starts = [
        match for match in starts_all
        if id(match) not in toc_starts and id(match) not in cross_starts
    ]
    standard_ends = [match for match in standard_ends if id(match) not in cross_ends]
    combined_ends = [match for match in combined_ends if id(match) not in cross_ends]
    ends = [match for match in ends_all if id(match) not in cross_ends]

    # Some compact filings expose only one Item 1 marker and place Item 1A very
    # close to it in a table of contents.  If the TOC heuristic removed every
    # otherwise credible start, restore the non-cross-reference starts and make
    # the fallback explicit in the audit warning rather than silently failing.
    used_toc_fallback = False
    if not starts:
        starts = [match for match in starts_all if id(match) not in cross_starts]
        used_toc_fallback = bool(starts)

    # Retain both uppercase and title-case headings.  Some filings use an
    # uppercase Item 1A in the table of contents but a title-case Item 2 at the
    # real section boundary; globally filtering by capitalization would discard
    # that valid endpoint.

    candidates: list[tuple[int, re.Match[str], re.Match[str], str]] = []
    short_candidates: list[tuple[int, re.Match[str], re.Match[str], str]] = []
    for start in starts:
        # The first credible following heading is the end of Item 1.  Choosing
        # the longest start/end pairing incorrectly absorbs Item 1A (Risk
        # Factors), which was the instability observed in repeated page-header
        # filings.
        eligible_ends = combined_ends if id(start) in combined_start_ids else standard_ends
        end = next((item for item in eligible_ends if item.start() > start.end()), None)
        if end is None:
            continue
        candidate = text[start.end():end.start()].strip()
        words = len(candidate.split())
        if min_words <= words <= max_words:
            candidates.append((words, start, end, candidate))
        elif 100 <= words < min_words:
            short_candidates.append((words, start, end, candidate))

    used_short_fallback = False
    if not candidates and short_candidates:
        # A few valid historical Item 1 sections are genuinely shorter than the
        # configured safety threshold.  Preserve them with an explicit audit
        # warning instead of silently omitting the filing.
        candidates = short_candidates
        used_short_fallback = True
    used_part_i_fallback = False
    if not candidates:
        # A few filings render the Item 1 label as an image and expose only
        # "PART I OVERVIEW" in visible text. Pair that structural marker with
        # the first later Item 1A/Item 2 heading. The TOC copy yields only a
        # handful of words, while the real Part I marker yields the full span.
        part_i_candidates: list[tuple[int, re.Match[str], re.Match[str], str]] = []
        for start in PART_I_BUSINESS_START.finditer(text):
            end = next((item for item in ends if item.start() > start.end()), None)
            if end is None:
                continue
            candidate = text[start.end():end.start()].strip()
            words = len(candidate.split())
            if 100 <= words <= max_words:
                part_i_candidates.append((words, start, end, candidate))
        if part_i_candidates:
            candidates = part_i_candidates
            used_part_i_fallback = True
    if not candidates:
        raise ValueError(
            f"No plausible Item 1 span for {source_file}: "
            f"usable_starts={len(starts)} usable_ends={len(ends)} "
            f"rejected_toc={len(toc_starts)} rejected_start_xref={len(cross_starts)} "
            f"rejected_end_xref={len(cross_ends)}"
        )

    words, start, end, section = max(candidates, key=lambda item: item[0])
    warnings: list[str] = []
    if used_toc_fallback:
        warnings.append("toc_start_fallback")
    if used_short_fallback:
        warnings.append(f"short_item1_under_min_{min_words}_words")
    if used_part_i_fallback:
        warnings.append("part_i_overview_fallback")
    if id(start) in combined_start_ids:
        warnings.append("combined_items_1_2")
    if words < warning_words:
        warnings.append(f"short_item1_under_{warning_words}_words")
    result = BoundaryResult(
        source_file=source_file,
        visible_words=len(text.split()),
        section_words=words,
        start_char=start.start(),
        end_char=end.start(),
        start_text=text[start.start():start.end() + 100],
        end_text=text[end.start():end.end() + 100],
        start_candidates=len(starts_all),
        end_candidates=len(ends_all),
        rejected_toc_starts=len(toc_starts),
        rejected_cross_reference_starts=len(cross_starts),
        rejected_cross_reference_ends=len(cross_ends),
        warning=";".join(warnings),
    )
    return section, result
