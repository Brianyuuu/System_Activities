# Dependency phrase extraction

This pipeline extracts candidate operational activities from the substantive
Item 1 (Business) section of each fiscal-2015 SEC 10-K. It uses spaCy dependency
parsing and supports three nested sensitivity modes.

- **Strict:** precision-oriented active-voice firm subjects and direct objects,
  with compact noun phrases.
- **Medium:** adds coordinated verbs and objects, passive clauses with an
  explicit firm agent, prepositional objects, complements, and longer noun
  phrases that retain mode, scope, negation, and other strategic modifiers.
- **Loose:** also retains firm-subject passives without explicit agents,
  copular/auxiliary constructions, a broader operational verb list, and
  borderline strategic or aspirational constructions.

Each output row contains `extraction_sensitivity`, a deterministic `rule_score`,
the rules triggered, normalized phrase, object, detected subject, source
sentence, and filing identifiers. Scores measure syntactic evidence and are not
probabilities. The evidence fields make every extraction auditable.

## Reproduce

From `Empirical_explore` on macOS or Linux:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r codes/requirements.txt
chmod +x codes/run_extraction.sh
codes/run_extraction.sh
```

The script reads `10k/*.htm` and creates:

- one file per company and mode, e.g. `tesla_loose_2015.csv`;
- pooled files for strict, medium, and loose modes; and
- `all_companies_combined_long_2015.csv`, which stacks all modes.

To run only one mode, append `--sensitivity strict`, `medium`, or `loose` to the
Python command in `run_extraction.sh`. The default is `all`.

## Important scope decisions

- The input is Item 1 only, bounded by Item 1A (Risk Factors).
- A table-of-contents match is rejected by selecting the longest plausible
  Item 1-to-Item 1A span.
- Medium and loose conjoined verbs inherit a qualifying firm subject.
- Medium and loose modes preserve coordinated objects and prepositional scope.
- Bare verbs are excluded except for verbs on the broadened loose activity list.
- This stage intentionally favors transparent syntactic rules. It does not yet
  classify phrases as activities versus resources, capabilities, or outcomes;
  that is a later cleaning and validation task.

## DeepSeek LLM extraction

`extract_llm_activities.py` is a separate, non-cleaning extraction stage. It
splits Item 1 into non-overlapping sentence-bounded chunks, calls DeepSeek in JSON
mode, and rejects any activity whose source sentence is not an exact sentence
from its input chunk. It writes company JSON and comparable long-format CSV
files to `llm_phrases/`, plus raw per-chunk responses and rejection decisions
under `llm_phrases/audit/` for restart and audit purposes.

The LLM pipeline has three nested sensitivity modes:

- **Strict:** explicitly stated, current operational activities under the base
  prompt. Existing validated strict responses are reused from cache.
- **Medium:** adds coordinated and passive constructions, prepositional
  activities, longer noun phrases, and value creation, delivery, coordination,
  and capture practices while preserving strategic modifiers.
- **Loose:** additionally admits textually explicit borderline operating,
  organizational, investment, expansion, planned, and continuing actions.

The generated sets are nested by construction (`strict ⊆ medium ⊆ loose`),
and every row includes `extraction_sensitivity`. Exact source-sentence matching
remains mandatory at every sensitivity. Company outputs follow the pattern
`tesla_deepseek_loose_2015.csv`; pooled mode files and a combined long-format
file are also produced.

The API key is never stored in code or output. Run:

```bash
export DEEPSEEK_API_KEY="your-key"
codes/run_llm_extraction.sh
```

The default endpoint is `https://api.deepseek.com` and the default model is
`deepseek-v4-pro` in non-thinking JSON mode. Override them with
`DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`. Cached
successful chunks are reused, so interrupted runs resume without paying for
completed requests.

## Canonical phrase organization and comparisons

Run `codes/run_phrase_comparisons.sh` to copy only sensitivity-labeled method
outputs into `phrases/dependency_phrases/` and `phrases/llm_phrases/`. Legacy
files remain in their original folders but are ignored. The script creates one
exact-key comparison per company and sensitivity in `phrases/comparison/`, plus
`comparison_summary.csv`.

Matching uses `(activity_phrase_clean, source_sentence)` only when both source
files contain `activity_phrase_clean`; otherwise it uses
`(activity_phrase, source_sentence)`. `method_match` is `dependency`, `llm`, or
`both`. Metadata from both matched rows is consolidated without discarding
method-specific fields.

## Medium cleaning and deduplication

The current Step 3 workflow is split into auditable stages rather than one
monolithic script. It uses only the four medium comparison CSVs in
`phrases/comparison/`.

Run the cleaning stages from `Empirical_explore`:

```bash
.venv/bin/python codes/clean_step1_no_bad_phrases.py
.venv/bin/python codes/deduplicate_same_sentence.py
```

`clean_step1_no_bad_phrases.py` normalizes phrase text, preserves strategic
modifiers such as negation and scope terms, and removes obvious bad extractions
such as single-token phrases, syntactic fragments, and phrases with too little
activity content. It writes outputs under
`phrases/clean_normalize/no_bad_phrases/`.

`deduplicate_same_sentence.py` compares phrases within the same firm, year, and
source sentence, then retains a representative phrase when multiple extractions
describe the same underlying activity. It writes outputs under
`phrases/clean_normalize/deduplication/`.

## DeepSeek semantic duplicate and actor labeling

Run:

```bash
export DEEPSEEK_API_KEY="your-key"
.venv/bin/python codes/deepseek_duplicate_actor_labels.py
.venv/bin/python codes/final_dedup_firm_only.py
```

`deepseek_duplicate_actor_labels.py` labels candidate phrase pairs as
`duplicate`, `related_but_distinct`, or `not_duplicate`, and identifies the
actor performing each activity. It writes semantic duplicate and actor audit
outputs under `phrases/clean_normalize/deduplication/` and
`phrases/clean_normalize/actor_identification/`.

`final_dedup_firm_only.py` keeps only focal-firm activities and removes the
remaining firm-level semantic duplicates. The final 522-row activity file is:

```text
phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv
```

## Embeddings and near-duplicate checks

Run:

```bash
.venv/bin/python codes/embed_and_detect_near_duplicates.py
```

This embeds the final firm-only activity phrases with
`sentence-transformers/all-mpnet-base-v2`, writes the reproducible row map and
embedding array under `embedding/`, and reports near-duplicate pairs across high
cosine-similarity thresholds. Near-duplicate pairs are audited but not collapsed
at this stage.

## Clustering grid, stability, and firm-activity matrices

Run:

```bash
.venv/bin/python codes/cluster_step4_method_grid.py
.venv/bin/python codes/cluster_step4_stability.py
.venv/bin/python codes/build_firm_activity_matrices.py
```

`cluster_step4_method_grid.py` clusters the 522 final firm-only activity phrases
with HDBSCAN, k-means, hierarchical Ward clustering, and BERTopic. Outputs are
written to `clustering/`; each run contains `assignments.csv`,
`cluster_summary.csv`, `representative_phrases.csv`, and `run_metadata.json`.

`cluster_step4_stability.py` runs stability checks against the original
clustering outputs: random seed reruns, leave-Tesla-out reclustering, and 80%
subsampling. ARI summaries are written to `stability/`.

`build_firm_activity_matrices.py` converts each clustering run into binary,
count, weighted-share, and TF-IDF-style firm-year x activity-cluster matrices
under `firm_act_matrix/`.

## Archived scripts

Older monolithic or exploratory scripts were moved to `codes/archive/`. They are
kept for provenance but are not part of the current reproducible workflow.
