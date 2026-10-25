# Reproducibility assessment — 2026-10-25

**The workflow is implementable and its packaged local logic is offline-tested. It is not yet certified as a fresh-checkout live reproduction of the historical research results, and production coverage is not complete.** These are separate claims.

## What is supplied

- The original repository history, exploratory code and tracked outputs, plus an archived copy of its original README.
- Source verification and primary-Item-1 release construction adapted from the deployed PARCC code, with local-original/hash support and explicit identity inputs.
- Deployed extraction stages, strict checkpoint runtime and prompts; isolated-run preparation and an industry export adapter.
- The portable GLM/Gemini assignment component, explicit recent/legacy recall policies, frozen code/input validation, and 19,587 parent definitions.
- A new portable implementation of the documented industry matrix method, explicit coverage and World contracts, embedding provenance, and workbook readback validation.
- Pinned direct dependencies, synthetic tests and a Linux CI workflow. Transitive/platform dependencies are resolved during installation; this is not a container image or a fully locked cross-platform environment.

`source_provenance.json` records source hashes and adaptations. `taxonomy_provenance.json` records the definition-only taxonomy projection. `release_manifest.json` records distributed file hashes. A source hash is provenance, not evidence of model accuracy.

## What the tests establish

The no-network suite uses a synthetic original filing and fake model responses. It checks:

1. Approved versus held source identity, primary Item 1 boundaries, immutable release verification, raw/code/output tamper rejection, and preservation of held-year coverage.
2. Phrase-column adapters and preservation of original source rows, duplicate/context grouping and the four-word boundary.
3. Actual embedding-recall/GLM-filter code with fake responses, persisted caches/usage ledger, completion gates, and a zero-new-call cached rerun.
4. Explicit top-1,000 versus legacy threshold semantics and ties, Gemini candidate preparation, response parsing, reciprocal intersection, and final source-row export.
5. Frozen scientific configuration rejection after edits.
6. Matrix coverage, source/World hashes, counts, adjacent-year entry/exit, missingness, scoped first observations, known cosine distances, tie ordering, zero centered vectors, uncertain embedding-intent blocking, and XLSX readback.
7. Python syntax and extraction executable imports/CLI help with installed dependencies.

The local release verification uses Python 3.14. CI targets Python 3.12/3.14 on Linux; check the repository's Actions result for the actual remote outcome. No real model inference, SEC download, Slurm submission, or paid matrix embedding is part of the offline suite. Synthetic stage receipts in tests are deliberately labeled test fixtures and do not certify extraction model quality.

## What is external or still unverified

- **Research inputs:** reviewed economic-entity history, accession mappings, raw originals, requested firm-year coverage, and the complete World membership corpus. The repo does not bundle private research snapshots or infer historical ticker identity automatically.
- **Access and service state:** PARCC allocation/network access; authorized API keys; model availability; provider quotas/rates; and scheduler capacity.
- **Scientific quality:** extraction's technical QA preserves semantic-review flags. No claim that every phrase or assignment is semantically correct, or that every possible activity is covered.
- **Live integration:** the new portable packaging and matrix builder need a bounded live pilot with real credentials before scaling. Historical live output validates only its original configuration and inputs.
- **Complete population coverage:** held, failed or unfinished reports remain missing. An entirely empty phrase dataset requires a separate reviewed zero-result handling step; the assignment adapter intentionally refuses an empty source CSV.
- **Historical equivalence:** switching from legacy uncapped recall to top-1,000 changes candidates. Do not pool results without recording the policy. Model aliases and services may change, and even temperature-zero inference may differ. The projected taxonomy has the same consumed definitions but a different file hash from the full historical catalog.
- **Presentation:** matrix numeric/cell validation is automated, visual rendering is not certified. The new builder implements the method, not historical workbook bytes.

## Production snapshot that motivated this update

Read-only observations on October 25, 2026, roughly 17:09–17:30 America/New_York (different stages were sampled at different times):

| Area | Observed state |
|---|---|
| Source verification queue | 110 completed queue entries, 13,111 approved reports; four failed entries and seven live workers still present |
| Banks/hardware extraction | 677/1,982 reports validated; 30,594 phrases; active array and one preserved failure |
| Retail held-report extraction | 6/178 validated; 549 phrases; active array |
| Latest banks B7 GLM | Completed screening: 859,140 candidates, 15,987 kept; not a final Gemini result |
| Latest hardware B6 GLM | Completed screening: 888,617 candidates, 25,988 kept; not a final Gemini result |
| Earlier airlines/entertainment/cars | Completed 19,587-parent Gemini audit evidence existed; historical matrices existed but were not newly validated as part of that inspection |

These are historical observations, not a live dashboard. The repository update neither stops nor restarts these jobs and does not overwrite their outputs. Completion of selected older industries does not establish completion of the current full pipeline population.

## Reproducing a result

For a defensible rerun, retain the repository commit, installed environment inventory, exact mapping/raw/coverage files, taxonomy bytes, policy/configuration, prompts, World corpus and all hashes. Keep the whole run, model/version/usage receipts, responses, checkpoint state, embedding caches and ledgers. Reuse compatible cached judgments to reproduce recorded outputs; a fresh model inference run is a new experiment, even with the same code.

Before declaring the full study complete, reconcile every target firm-year, resolve or document held/failed sources, complete extraction and reciprocal assignment gates, build each industry's matrix against an explicitly frozen World scope, and review both scientific quality and presentation. There is no single scheduler status or flag that substitutes for those artifacts.
