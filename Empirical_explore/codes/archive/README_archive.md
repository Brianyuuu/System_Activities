# Archived Scripts

These scripts are older monolithic or exploratory versions of the activity
extraction pipeline. They are kept for provenance, but they are not part of the
current reproducible workflow.

Archived files:

- `clean_normalize_medium.py`
- `run_clean_normalize_medium.sh`
- `embed_final_activities.py`
- `cluster_activity_taxonomy.py`
- `run_cluster_activity_taxonomy.sh`
- `cluster_step4_emergent_categories.py`

The current workflow uses the stepwise scripts in `codes/`:

1. `extract_dependency_phrases.py`
2. `extract_llm_activities.py`
3. `build_phrase_comparisons.py`
4. `clean_step1_no_bad_phrases.py`
5. `deduplicate_same_sentence.py`
6. `deepseek_duplicate_actor_labels.py`
7. `final_dedup_firm_only.py`
8. `embed_and_detect_near_duplicates.py`
9. `cluster_step4_method_grid.py`
10. `cluster_step4_stability.py`
11. `build_firm_activity_matrices.py`
