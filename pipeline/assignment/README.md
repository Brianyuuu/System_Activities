# Phrase-to-parent assignment

This is the portable assignment component of [System Activities](../../README.md). It derives from the October 22, 2026 PARCC package, with explicit recall policy selection, frozen input/code validation, and repository-wide installation.

From the repository root, follow the [runbook](../../docs/RUNBOOK.md). Install the root `requirements.txt`; no separate bootstrap or private source snapshot is required.

`prepare_run.py` accepts `final_phrase`, `atom_mechanism_phrase`, or `activity_phrase` (in that precedence), `original_sentence`, `year`, and `ticker` or `firm_id`. Use `--phrase-column` for another explicit column. Original columns and original row numbers remain available in the final export. Existing verification status must agree with the verified-input contract. `--confirm-verified-source` records operator review; it does not perform source verification.

The default frozen taxonomy contains 19,587 parent definitions. Its distributed projection omits historical memberships, which must be supplied separately when building World metrics. It preserves the exact ID, label, inclusion, and exclusion fields consumed by this component.

`--recall-policy top1000` selects score > 0.33 intersect global top 1,000; `legacy` selects all scores >= 0.33. Parent-index order breaks ties. GLM-kept candidates union the mandatory embedding top 30 proceed to Gemini. Full final assignments require reciprocal Gemini completion; GLM-only results remain provisional.

Stage scripts support `offline`, `smoke`, `glm`, and `gemini`. `smoke` makes one real GLM call. `glm` and `gemini` make live API calls. The Linux stage wrapper takes a lock and respects the run's prepared worker count; environment overrides are operational controls. Scheduler account and QOS are explicit arguments to `make_sbatch.py`; generation never submits a job.

Keep the credential JSON outside the repository, mode 0600, addressed by `PIPELINE_SECRET_FILE`. Field names are shown in `templates/credentials.example.json`. Do not copy live caches, ledgers or original source data into the repository. Resume compatible runs in place instead of replaying paid calls from a fresh directory.

`MANIFEST.json` verifies the distributed component; `scripts/verify_bundle.py` also checks prompt parity. `tests/test_input_adapter.py` checks input contracts, and `tests/test_portable_pipeline.py` exercises real local recall/filter/review/export logic with synthetic responses, blocked network, cached resume, threshold boundaries and configuration tamper rejection. Passing these tests does not certify live endpoints or model quality.
