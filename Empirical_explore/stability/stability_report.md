# Clustering stability report

Baseline clustering folder: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/clustering`

Stability output folder: `/Users/brian/Dropbox/Wharton/SystemAct/Empirical_explore/stability`

Input: 522 final firm-only deduplicated activity phrases.

Ticker counts in the 522-phrase input:

- TSLA: 224
- LUV: 130
- F: 88
- DAL: 80

## Stability checks performed

1. Random seed reruns
   - Seeds: 7, 13, 29
   - K-means and BERTopic are seed-sensitive.
   - HDBSCAN and Ward are deterministic, so their seed ARI is expected to be 1.0.

2. Leave Tesla out
   - Excluded TSLA.
   - Reclustering rows from LUV, F, and DAL only.
   - Compared against the original clustering restricted to those same non-TSLA rows.
   - Compared rows: 298.

3. 80% subsampling
   - Subsample fraction: 80%.
   - Seeds: 101, 202, 303, 404, 505.
   - Each subsample has 418 rows.
   - Compared against the original clustering restricted to each sampled row set.

## Method-level ARI summary

| test_type | method | n_tests | mean_ari | median_ari | min_ari | max_ari |
|---|---|---:|---:|---:|---:|---:|
| leave_tesla_out | bertopic | 1 | 0.455 | 0.455 | 0.455 | 0.455 |
| leave_tesla_out | hdbscan | 5 | 0.538 | 0.457 | 0.000 | 0.956 |
| leave_tesla_out | hierarchical_ward | 4 | 0.612 | 0.630 | 0.492 | 0.695 |
| leave_tesla_out | kmeans | 4 | 0.419 | 0.424 | 0.302 | 0.525 |
| random_seed | bertopic | 3 | 0.494 | 0.481 | 0.448 | 0.553 |
| random_seed | hdbscan | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_seed | hierarchical_ward | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_seed | kmeans | 12 | 0.466 | 0.424 | 0.379 | 0.658 |
| subsample_80pct | bertopic | 5 | 0.403 | 0.395 | 0.351 | 0.445 |
| subsample_80pct | hdbscan | 25 | 0.532 | 0.708 | -0.097 | 0.929 |
| subsample_80pct | hierarchical_ward | 20 | 0.620 | 0.650 | 0.382 | 0.782 |
| subsample_80pct | kmeans | 20 | 0.434 | 0.407 | 0.306 | 0.626 |

## Interpretation

The most stable family across perturbation tests is hierarchical Ward. It is deterministic under seed changes and retains the highest average ARI under leave-Tesla-out and 80% subsampling.

HDBSCAN is deterministic under seed changes, but its stability varies substantially when the sample changes. Some parameter settings produce high ARI, while others collapse or reassign many points, especially when Tesla is excluded or the sample is reduced.

K-means is moderately unstable under seed changes and data perturbations. This is expected because high-dimensional embeddings, many clusters, and small clusters make centroid assignment sensitive.

BERTopic shows moderate-to-low stability. This is also expected because the integrated UMAP + HDBSCAN topic pipeline is sensitive to random state and corpus composition.

Core output files:

- `ari_against_original_all_runs.csv`
- `ari_summary_by_test_method_run.csv`
- `ari_summary_by_test_method.csv`
