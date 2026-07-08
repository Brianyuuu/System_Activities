#!/usr/bin/env python3
"""Extract firm-subject activity phrases from SEC 10-K Item 1 sections.

The script reads original SEC HTML filings, identifies the substantive Item 1
(Business) section, parses it with spaCy, and emits auditable long-format CSVs.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import spacy
from lxml import html as lxml_html
from spacy.tokens import Doc, Token


# Some inline-XBRL filings render ITEM as "I TEM" because the initial letter
# and remainder have separate styling. Optional whitespace handles both forms.
ITEM1_START = re.compile(r"\bi\s*tem\s+1\s*[.\-–—:]?\s*business\b", re.I)
ITEM1_END = re.compile(r"\bi\s*tem\s+1\s*a\s*[.\-–—:]?\s*risk\s+factors\b", re.I)
SPACE = re.compile(r"\s+")
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
COMPLEMENT_DEPS = {"xcomp", "ccomp"}
PHRASE_DEPS = OBJECT_DEPS | COMPLEMENT_DEPS | {"prt", "prep", "agent", "acomp"}
Sensitivity = Literal["strict", "medium", "loose"]
SENSITIVITIES: tuple[Sensitivity, ...] = ("strict", "medium", "loose")

# The list affects confidence and permits otherwise-bare candidates in loose
# mode. Dependency evidence, rather than dictionary membership, remains the
# primary inclusion criterion in strict and medium modes.
CORE_ACTIVITY_VERBS = {
    "acquire", "assemble", "build", "buy", "design", "develop", "distribute",
    "employ", "finance", "lease", "manufacture", "market", "operate", "produce",
    "provide", "purchase", "research", "sell", "serve", "service", "source",
    "support", "train", "transport", "use",
}
LOOSE_ACTIVITY_VERBS = CORE_ACTIVITY_VERBS | {
    "adopt", "allow", "continue", "control", "coordinate", "create", "deliver",
    "establish", "expand", "focus", "implement", "intend", "invest", "maintain",
    "manage", "offer", "own", "plan", "seek", "specialize", "target", "work",
}
ASPIRATIONAL_VERBS = {"aim", "believe", "expect", "hope", "intend", "plan", "seek"}
NP_MODIFIER_DEPS = {"amod", "compound", "nummod", "poss", "case", "det", "neg", "quantmod"}


@dataclass(frozen=True)
class Filing:
    company: str
    firm_id: str
    year: str
    accession: str
    aliases: tuple[str, ...]
    path: Path


FILING_PATTERN = re.compile(
    r"^(?P<firm>tesla|ford|southwest|delta)_(?P<year>\d{4})_10-k_"
    r"(?P<accession>[\d-]+)\.htm$",
    re.I,
)

FIRM_META = {
    "tesla": ("Tesla Motors, Inc.", "1318605", ("tesla", "tesla motors")),
    "ford": ("Ford Motor Company", "37996", ("ford", "ford motor", "ford motor company")),
    "southwest": (
        "Southwest Airlines Co.",
        "92380",
        ("southwest", "southwest airlines", "southwest airlines co"),
    ),
    "delta": ("Delta Air Lines, Inc.", "27904", ("delta", "delta air lines")),
}

FIELDNAMES = [
    "firm_id",
    "company",
    "year",
    "accession",
    "source_method",
    "extraction_sensitivity",
    "rule_score",
    "rules_triggered",
    "activity_phrase",
    "verb",
    "object",
    "subject",
    "subject_dependency",
    "sentence_index",
    "source_sentence",
    "source_file",
]


def normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    return SPACE.sub(" ", value).strip()


def html_to_text(path: Path) -> str:
    parser = lxml_html.HTMLParser(encoding="utf-8", recover=True)
    root = lxml_html.fromstring(path.read_bytes(), parser=parser)
    for node in root.xpath("//script|//style|//noscript"):
        node.drop_tree()
    return normalize_text(" ".join(root.itertext()))


def extract_item1(text: str) -> str:
    """Select the longest plausible Item 1-to-Item 1A span.

    SEC filings commonly repeat section names in a table of contents. Pairing
    every Item 1 marker with the next Item 1A marker and selecting the longest
    plausible candidate reliably rejects the short TOC entry.
    """
    starts = list(ITEM1_START.finditer(text))
    ends = list(ITEM1_END.finditer(text))
    candidates: list[str] = []
    for start in starts:
        end = next((m for m in ends if m.start() > start.end()), None)
        if end is None:
            continue
        candidate = text[start.end() : end.start()].strip()
        word_count = len(candidate.split())
        if 500 <= word_count <= 100_000:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            f"Could not isolate a plausible Item 1 section "
            f"({len(starts)} starts, {len(ends)} ends)."
        )
    return max(candidates, key=lambda value: len(value.split()))


def filing_from_path(path: Path) -> Filing:
    match = FILING_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filing filename: {path.name}")
    key = match.group("firm").lower()
    company, firm_id, aliases = FIRM_META[key]
    return Filing(
        company=company,
        firm_id=firm_id,
        year=match.group("year"),
        accession=match.group("accession"),
        aliases=aliases,
        path=path,
    )


def token_text(token: Token) -> str:
    return normalize_text(token.text).lower().strip(".,;:()[]{}")


def is_firm_subject(subject: Token, aliases: tuple[str, ...]) -> bool:
    phrase = normalize_text(" ".join(t.text for t in subject.subtree)).lower()
    head = token_text(subject)
    if head in {"we", "company", "registrant"}:
        return True
    if any(alias in phrase for alias in aliases):
        return True
    # Covers constructions such as "our operations provide ...".
    return any(
        child.dep_ == "poss" and token_text(child) in {"our", "company's"}
        for child in subject.children
    )


def explicit_subject(verb: Token) -> Token | None:
    return next(
        (child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass"}),
        None,
    )


def active_subject(verb: Token) -> Token | None:
    return next((child for child in verb.children if child.dep_ == "nsubj"), None)


def passive_patient(verb: Token) -> Token | None:
    return next((child for child in verb.children if child.dep_ == "nsubjpass"), None)


def inherited_subject(verb: Token) -> Token | None:
    current = verb
    while current.dep_ == "conj" and current.head != current:
        current = current.head
        subject = explicit_subject(current)
        if subject is not None:
            return subject
    return None


def subtree_tokens(root: Token) -> list[Token]:
    return sorted(root.subtree, key=lambda token: token.i)


def firm_agent(verb: Token, aliases: tuple[str, ...]) -> Token | None:
    """Return the firm token in a passive `by us/the company` agent phrase."""
    for child in verb.children:
        if child.dep_ not in {"agent", "prep"} or child.lower_ != "by":
            continue
        for candidate in child.subtree:
            if candidate.dep_ in {"pobj", "obj"} and is_firm_subject(candidate, aliases):
                return candidate
    return None


def compact_np(root: Token, sensitivity: Sensitivity) -> list[Token]:
    """Expand noun phrases while preserving mode, scope, and negation modifiers."""
    if sensitivity == "strict":
        selected = {root.i: root}
        for token in root.subtree:
            if token.dep_ in NP_MODIFIER_DEPS and token.head.i in selected:
                selected[token.i] = token
        return [selected[i] for i in sorted(selected)]

    # Medium/loose retain longer NPs, coordinated objects, and attached scope
    # phrases but stop at finite clauses that commonly contain unrelated claims.
    blocked = {"relcl", "advcl", "ccomp", "parataxis"}
    selected: list[Token] = []
    for token in root.subtree:
        cursor = token
        reject = False
        while cursor != root:
            if cursor.dep_ in blocked:
                reject = True
                break
            cursor = cursor.head
        if not reject and not token.is_punct:
            selected.append(token)
    return sorted(selected, key=lambda token: token.i)


def phrase_for_verb(
    verb: Token,
    sensitivity: Sensitivity,
    passive_object: Token | None = None,
) -> tuple[str, str, set[str]]:
    rules: set[str] = set()
    roots: list[Token] = []
    object_roots = [child for child in verb.children if child.dep_ in OBJECT_DEPS]
    if passive_object is not None:
        object_roots.append(passive_object)
        rules.add("passive_patient_as_object")
    if object_roots:
        roots.extend(object_roots)
        rules.add("direct_object")

    if sensitivity in {"medium", "loose"}:
        preps = [child for child in verb.children if child.dep_ == "prep" and child.lower_ != "by"]
        complements = [child for child in verb.children if child.dep_ in COMPLEMENT_DEPS | {"acomp", "prt"}]
        if preps:
            roots.extend(preps)
            rules.add("prepositional_object")
        if complements:
            roots.extend(complements)
            rules.add("verbal_or_adjective_complement")

    tokens: list[Token] = []
    object_tokens: list[Token] = []
    for root in roots:
        expanded = compact_np(root, sensitivity)
        tokens.extend(expanded)
        if root in object_roots:
            object_tokens.extend(expanded)
        if sensitivity != "strict" and any(token.dep_ == "conj" for token in expanded):
            rules.add("coordinated_objects")
        if sensitivity != "strict" and len(expanded) > 1:
            rules.add("expanded_noun_phrase")

    unique = {token.i: token for token in tokens if not token.is_punct}
    ordered = [unique[index] for index in sorted(unique)]
    if not ordered and not (sensitivity == "loose" and verb.lemma_.lower() in LOOSE_ACTIVITY_VERBS):
        return "", "", rules
    if not ordered:
        rules.add("loose_activity_verb")

    verb_form = verb.lemma_.lower() if verb.lemma_ != "-PRON-" else verb.lower_
    phrase = normalize_text(" ".join([verb_form] + [token.text for token in ordered]))
    object_text = normalize_text(" ".join(token.text for token in object_tokens if not token.is_punct))
    return phrase, object_text, rules


def score_candidate(verb: Token, rules: set[str]) -> float:
    score = 0.0
    if "explicit_firm_subject" in rules:
        score += 0.38
    if "coordinated_verb" in rules:
        score += 0.28
    if "passive_firm_agent" in rules:
        score += 0.38
    if "loose_firm_passive_subject" in rules:
        score += 0.20
    if "direct_object" in rules:
        score += 0.28
    if "prepositional_object" in rules:
        score += 0.12
    if "verbal_or_adjective_complement" in rules:
        score += 0.10
    if verb.lemma_.lower() in CORE_ACTIVITY_VERBS:
        score += 0.12
        rules.add("core_activity_verb")
    elif verb.lemma_.lower() in LOOSE_ACTIVITY_VERBS:
        score += 0.06
        rules.add("broadened_activity_verb")
    if verb.pos_ == "AUX":
        score -= 0.12
        rules.add("auxiliary_or_copular")
    if verb.lemma_.lower() in ASPIRATIONAL_VERBS:
        score -= 0.10
        rules.add("borderline_aspirational_verb")
    return round(max(0.0, min(1.0, score)), 2)


def activity_rows(
    doc: Doc, filing: Filing, sensitivity: Sensitivity
) -> Iterable[dict[str, str | int | float]]:
    seen: set[tuple[str, str]] = set()
    for sentence_index, sentence in enumerate(doc.sents, start=1):
        source_sentence = normalize_text(sentence.text)
        if len(source_sentence) < 20:
            continue
        for verb in sentence:
            if verb.pos_ not in ({"VERB"} if sensitivity == "strict" else {"VERB", "AUX"}):
                continue
            rules: set[str] = set()
            subject = active_subject(verb)
            passive_object: Token | None = None

            if subject is not None and is_firm_subject(subject, filing.aliases):
                rules.add("explicit_firm_subject")
            else:
                subject = None

            if subject is None and sensitivity in {"medium", "loose"}:
                inherited = inherited_subject(verb)
                if inherited is not None and is_firm_subject(inherited, filing.aliases):
                    subject = inherited
                    rules.add("coordinated_verb")

            if subject is None and sensitivity in {"medium", "loose"}:
                agent = firm_agent(verb, filing.aliases)
                patient = passive_patient(verb)
                if agent is not None and patient is not None:
                    subject = agent
                    passive_object = patient
                    rules.add("passive_firm_agent")

            if subject is None and sensitivity == "loose":
                patient = passive_patient(verb)
                if patient is not None and is_firm_subject(patient, filing.aliases):
                    subject = patient
                    rules.add("loose_firm_passive_subject")

            if subject is None:
                continue
            phrase, object_text, phrase_rules = phrase_for_verb(
                verb, sensitivity, passive_object=passive_object
            )
            rules.update(phrase_rules)
            if not phrase:
                continue
            score = score_candidate(verb, rules)
            threshold = {"strict": 0.66, "medium": 0.50, "loose": 0.20}[sensitivity]
            if score < threshold:
                continue
            key = (phrase.lower(), source_sentence.lower())
            if key in seen:
                continue
            seen.add(key)
            yield {
                "firm_id": filing.firm_id,
                "company": filing.company,
                "year": filing.year,
                "accession": filing.accession,
                "source_method": "spacy_dependency",
                "extraction_sensitivity": sensitivity,
                "rule_score": f"{score:.2f}",
                "rules_triggered": "|".join(sorted(rules)),
                "activity_phrase": phrase,
                "verb": verb.lemma_.lower(),
                "object": object_text,
                "subject": normalize_text(" ".join(t.text for t in subject.subtree)),
                "subject_dependency": subject.dep_,
                "sentence_index": sentence_index,
                "source_sentence": source_sentence,
                "source_file": filing.path.name,
            }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument(
        "--sensitivity",
        choices=["all", *SENSITIVITIES],
        default="all",
        help="Extraction sensitivity; 'all' writes all three modes (default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*_10-k_*.htm"))
    if not paths:
        raise SystemExit(f"No filing HTML files found in {args.input_dir}")

    nlp = spacy.load(args.model, disable=["ner"])
    nlp.max_length = 5_000_000
    modes: tuple[Sensitivity, ...] = (
        SENSITIVITIES if args.sensitivity == "all" else (args.sensitivity,)
    )
    by_mode: dict[Sensitivity, list[dict[str, str | int | float]]] = {
        mode: [] for mode in modes
    }

    for path in paths:
        filing = filing_from_path(path)
        item1 = extract_item1(html_to_text(path))
        doc = nlp(item1)
        firm_slug = path.name.split("_", 1)[0].lower()
        counts: list[str] = []
        for mode in modes:
            rows = list(activity_rows(doc, filing, mode))
            output = args.output_dir / f"{firm_slug}_{mode}_{filing.year}.csv"
            write_csv(output, rows)
            by_mode[mode].extend(rows)
            counts.append(f"{mode}={len(rows):,}")
        print(f"{filing.company}: {len(item1.split()):,} Item 1 words; " + ", ".join(counts))

    combined: list[dict[str, str | int | float]] = []
    for mode in modes:
        rows = by_mode[mode]
        write_csv(args.output_dir / f"all_companies_{mode}_2015.csv", rows)
        combined.extend(rows)
    write_csv(args.output_dir / "all_companies_combined_long_2015.csv", combined)
    print(f"Combined long format: {len(combined):,} rows from {len(paths)} filings")


if __name__ == "__main__":
    main()
