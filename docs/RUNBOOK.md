# Runbook

Run commands from the repository root with its virtual environment activated. Keep originals in an external protected location or ignored `data/`, and generated runs under ignored `runs/`. Use full absolute paths for deployment. New runs must have new directories. Prepare on the final machine/path, since manifests freeze input locations. A macOS virtual environment cannot be copied to PARCC Linux.

The preparation and `--offline` commands below do not make model calls. The extraction execution, `glm`, `gemini`, and matrix `embed` commands do. There is no automatic scheduler submission or automatic chain starting paid stages.

## 1. Verify sources and freeze Item 1

Prepare the mapping described in [CONTRACTS.md](CONTRACTS.md). You must have reviewed historical entity identity and fiscal-year evidence; the confirmation flag records that review. For remote SEC acquisition, set a descriptive `SEC_USER_AGENT` with your own contact address. Local originals with matching supplied SHA-256 values do not require network access.

```bash
python scripts/prepare_extraction.py \
  --mapping /ABSOLUTE/PATH/accession_mapping.csv \
  --output runs/extraction_01 --confirm-identity
python scripts/run_extraction.py --run runs/extraction_01 --index 0 --offline
```

Inspect `runs/extraction_01/audits/item1_only_build.json`, `outputs/coverage_inventory.csv`, and source evidence packets. Indexes are zero-based rows of `inputs/releases/v1/approved_manifest.csv`. Held reports never enter that manifest. An empty approved release has no runnable index. Prepare a corrected new release after resolving source evidence; do not edit a frozen release to bypass validation.

## 2. Extract and export phrases

Set `LITELLM_API_KEY` through your protected runtime environment. This is the PARCC GLM credential, not an OpenAI key. `LITELLM_BASE_URL` optionally overrides the configured PARCC host and is recorded operationally by your deployment; use only an authorized equivalent endpoint. The fixed model is `parcc/GLM`.

```bash
python scripts/run_extraction.py --run runs/extraction_01 --index 0
```

Run each approved index once, normally in a bounded Slurm array. There are eight concurrent API workers per filing, so array parallelism multiplies endpoint load. The wrapper verifies frozen code, config, source and Item 1 hashes, acquires a filing lock, and preserves strict stage receipts. A previously completed receipt is reused only after output verification. Existing failed/running receipts require checkpoint review; they are not deleted or automatically restarted. Inspect the receipt and its `_strict_stage_state` logs, reconcile any uncertain calls, and preserve the audit trail before an explicitly reviewed recovery.

Export only after all approved reports for the selected source industry are complete:

```bash
python scripts/export_phrases.py --run runs/extraction_01 \
  --industry "Banks" --output runs/banks_phrases
```

The source industry is its exact mapping label. Export retains held years in `firm_years.csv` as incomplete. If you need target years outside the original mapping, add them explicitly to a new coverage manifest with `complete=0` and document the scope. Do not call missing extraction a confirmed negative.

## 3. Prepare and run GLM assignment

```bash
python pipeline/assignment/scripts/prepare_run.py \
  --source runs/banks_phrases/phrases.csv \
  --industry banks --identity-version YOUR_REVIEWED_IDENTITY_VERSION \
  --output runs/banks_assignment --workers 8 --budget-cap-usd 25 \
  --recall-policy top1000 --confirm-verified-source
python runs/banks_assignment/work/run_glm_only.py --offline
```

`25` is an illustrative budget ceiling, not a cost estimate. Size your pilot, then choose a budget for your actual data. This cap uses bundled estimated rates for retrieval embeddings and Gemini assignment batches. It does not cover extraction, internal GLM dollar cost, or later matrix embeddings. Rates and available models must be checked against your account before production use. Existing uncertain reservations remain committed; they are not a guarantee of provider billing reconciliation.

Create a credentials JSON **outside this repository**, following `pipeline/assignment/templates/credentials.example.json`; restrict it to mode 0600. Set its location without printing the contents:

```bash
export PIPELINE_SECRET_FILE=/ABSOLUTE/PRIVATE/PATH/credentials.json
export PYTHON_BIN="$(pwd)/.venv/bin/python"
bash pipeline/assignment/scripts/run_stage.sh runs/banks_assignment glm
```

The shell stage wrapper requires Linux `flock` (available on PARCC). For local offline checks use the Python commands. The pipeline preserves normalized units, embeddings, successful model responses, a SQLite usage/reservation ledger and checkpoints. Resume the same compatible run in place. Changed scientific inputs, prompts or configuration require a new run; the offline gate rejects frozen-file changes. Worker count is operational and may be changed via the worker-target file or environment.

A complete GLM screen has no pending/deferred pairs and passes `validate_glm_complete.py`. GLM-kept pairs are provisional, not final taxonomy assignments.

## 4. Gemini reciprocal audit and final export

Ensure the external credential JSON contains `GEMINI_API_KEY`, then:

```bash
bash pipeline/assignment/scripts/run_stage.sh runs/banks_assignment gemini
```

This stage validates the completed GLM screen, forms the kept-union-top30 candidate set, checks model batch support, submits/resumes Gemini parent-first and reciprocal batches, and exports source-row assignments. Paid batch submission intents and job names are persisted before/after submission. An uncertain submission must be reconciled rather than blindly resubmitted. Review the batch state and ledger when a run stops.

Do not build final matrices from `glm_kept_candidates.csv`. Require `runs/banks_assignment/outputs/pipeline/final/validation.json` with `complete=true` and `unresolved=0`. Unmatched phrases remain in `all_assigned.csv` with a resolved rejection/no-candidate status.

## 5. Declare World, embed labels, and build the industry matrix

Gather the completed assignment runs that define your comparison corpus. Repeated `--run` arguments must use the same frozen taxonomy. Include other completed industries when that is the intended meaning of World:

```bash
python scripts/build_world.py \
  --run runs/banks_assignment \
  --output runs/world_01 \
  --scope-description "Completed banks run only; no claims about other industries"
python scripts/matrix.py prepare \
  --run runs/banks_assignment \
  --coverage runs/banks_phrases/firm_years.csv \
  --world runs/world_01/world.csv --world-metadata runs/world_01/world.json \
  --output runs/banks_matrix
```

Alternatively supply a reviewed current-catalog-membership World CSV and matching provenance sidecar as specified in the contract. Historical membership data are not in the bundled taxonomy projection.

The payload freezes exact labels, coverage, metrics, source hashes and the semantic embedding contract. Set `GEMINI_API_KEY` in the environment for the next step. Choose an explicit maximum number of new labels after inspecting `payload.json`:

```bash
python scripts/matrix.py embed --directory runs/banks_matrix --max-new-labels 100
python scripts/matrix.py build --directory runs/banks_matrix
```

`100` is an example request-volume guard. If more labels are missing, the command fails before any calls. This is a separate paid embedding stage outside the assignment dollar cap. Cached vectors are bound to model, dimension, prefix and normalized label. Uncertain `.intent` files block overlapping retries until reconciled. Never delete them simply to force a retry. Use one embedding controller per matrix directory.

Outputs are `matrix_banks.xlsx`, `firm_year_panel.csv`, and `validation.json`. Numeric/cell checks are automated; visually review the workbook separately before sharing it. The builder refuses to overwrite an existing workbook. Preserve the payload, vectors/cache, validation and World metadata alongside published research results.

## PARCC deployment

Use your approved VAST project location and create the environment on Linux. Perform substantial extraction/embedding/matrix computation inside a compute allocation. Obtain current account, partition, QOS and walltime limits from your allocation; this repo does not hardcode the author's account or submit jobs.

Generate an assignment job at its final deployed location:

```bash
python pipeline/assignment/scripts/make_sbatch.py \
  --run runs/banks_assignment --stage glm \
  --account YOUR_ACCOUNT --qos YOUR_QOS --partition genoa-std-mem \
  --secret-file "$PIPELINE_SECRET_FILE" --cpus 8 --mem 44G --time 12:00:00
```

Inspect `runs/banks_assignment/glm.sbatch`, then submit it yourself with `sbatch` when ready. Generate a `--stage gemini` job after GLM completion. CPU/API workflows do not inherently need a GPU.

A minimal extraction array script can invoke the repository wrapper:

```bash
#!/usr/bin/env bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%2
#SBATCH --no-requeue
set -euo pipefail
umask 077
cd /ABSOLUTE/DEPLOYED/System_Activities
# Load LITELLM_API_KEY from your protected runtime environment here.
exec .venv/bin/python scripts/run_extraction.py --run /ABSOLUTE/RUN/extraction_01
```

Replace the illustrative `0-9` with exactly `0..approved_reports-1`, and choose parallelism for your endpoint quota. Never use a guessed array range. Monitor `squeue -u "$USER"`, then `sacct -j JOB_ID` and each stage's receipts/logs; scheduler exit alone does not prove scientific completion.
