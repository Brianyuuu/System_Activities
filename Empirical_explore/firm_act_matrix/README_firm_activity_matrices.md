# Step 6 firm-year × activity-cluster matrices

Input clustering folder: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/clustering`

Output folder: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/firm_act_matrix`

For every clustering run, this folder contains four wide firm-year × cluster matrices:

- `binary_matrix.csv`: 1 if the firm-year mentions at least one activity in the cluster, otherwise 0.
- `count_matrix.csv`: number of activity phrases from the firm-year assigned to the cluster.
- `weighted_share_matrix.csv`: count normalized by the firm-year's total non-noise activity mentions.
- `weighted_tfidf_matrix.csv`: count multiplied by `log((1+n_firmyears)/(1+df_cluster)) + 1`.

Each run folder also contains:

- `firm_year_cluster_counts_long_with_noise.csv`: long audit file, including `NOISE` when present.
- `cluster_dictionary.csv`: cluster columns, document frequency, IDF, and cluster summary metadata when available.
- `matrix_metadata.json`: run-level matrix metadata.

`NOISE`/unclustered labels are excluded from the wide matrices because they are not activity clusters, but they are retained in the long audit file.
