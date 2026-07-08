# Clustering stability checks

Baseline clustering folder: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/clustering`

Input phrases: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv`

Embeddings: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/embedding/firm_only_deduplicated_mpnet_embeddings.npy`

Stability checks:

1. `random_seed/`: reruns with seeds [7, 13, 29]. K-means and BERTopic are seed-sensitive; HDBSCAN and Ward are deterministic but rerun for comparison.
2. `leave_tesla_out/`: excludes TSLA and reclusters the other three firms.
3. `subsample_80pct/`: reclusters 80% phrase samples for seeds [101, 202, 303, 404, 505].

ARI outputs:

- `ari_against_original_all_runs.csv`: one row per stability run vs original run.
- `ari_summary_by_test_method_run.csv`: ARI summarized by test/method/run.
- `ari_summary_by_test_method.csv`: compact method-level summary.
