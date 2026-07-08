# SystemAct

Exploratory pipeline for extracting, cleaning, clustering, and evaluating firm
activity-system phrases from SEC 10-K Item 1 business descriptions.

The active workflow lives in `Empirical_explore/`. It combines rule-based
dependency parsing, LLM-assisted extraction and cleaning, semantic
deduplication, activity clustering, stability checks, and firm-year by activity
matrix construction.

## Repository layout

- `Empirical_explore/codes/`: reproducible scripts and workflow notes.
- `Empirical_explore/10k/`: small sample 2015 10-K HTML inputs.
- `Empirical_explore/dependency_phrases/`: rule-based extraction outputs.
- `Empirical_explore/llm_phrases/`: LLM extraction outputs.
- `Empirical_explore/phrases/`: cleaned and comparison phrase files.
- `Empirical_explore/clustering/`: activity taxonomy clustering outputs.
- `Empirical_explore/stability/`: clustering stability checks.
- `Empirical_explore/firm_act_matrix/`: firm-year by activity-cluster matrices.

## Reproduce

From `Empirical_explore`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r codes/requirements.txt
chmod +x codes/*.sh
codes/run_extraction.sh
```

For LLM-assisted stages, provide the API key through the environment rather than
storing it in the repository:

```bash
export DEEPSEEK_API_KEY="your-key"
codes/run_llm_extraction.sh
```

See `Empirical_explore/codes/README.md` for the full step-by-step workflow.

## Notes

This repository intentionally excludes local virtual environments, API keys,
large archives, generated binary arrays, and reading/reference PDFs or Word
documents. Regenerate those artifacts locally as needed.
