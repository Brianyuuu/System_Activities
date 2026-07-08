
## Pipeline

`10k/` -> `dependency_phrases/` / `llm_phrases/` -> `phrases/comparison/` -> `phrases/clean_normalize/` -> `embedding/` -> `clustering/` -> `stability/` -> `firm_act_matrix/`

###  Phrases Extraction 

We begin with two small within-industry contrasts: Tesla versus Ford in automobiles, and Southwest versus Delta in airlines. This design allows us to test whether the extraction pipeline can recover known differences between distinctive and traditional activity systems within each industry.

We extracted firm activities from 2015 10-K Item 1 descriptions using two methods. First, a spaCy dependency parser identified firm-subject verb phrases and their objects. Second, DeepSeek extracted explicitly stated activities from sentence-bounded text chunks; every accepted activity required an exact supporting sentence. Both methods used strict, medium, and loose sensitivity levels. Across the four firms, dependency parsing produced 256, 407, and 560 phrases, while DeepSeek produced 188, 516, and 650 phrases, respectively. Exact phrase overlap was low because the methods worded and divided activities differently.

Among the three sensitivity levels, the medium version best matches our working definition of firm activities and provides the best balance between precision and completeness. The strict version mainly captures very short verb-object phrases, which are often precise but difficult to interpret as full activities. The loose version captures more potentially useful activities, but it also turns some long sentences into phrases and introduces more irrelevant extractions. For this pilot, we use the medium version, while noting that the loose version may be worth re-filtering later.

Dependency parsing is transparent and gives us syntactically grounded verb-object phrases. However, it can lose crucial modifiers, including negation, which may reverse the meaning of an activity.

>  Southwest does not impose additional fees…

is extracted as

> additional fees

It can also fail when sentence structure becomes complex, especially when the activity depends on long-distance modifiers, coordination, or embedded clauses.

LLM generally does a better job, but faces the problem of duplication, i.e. it may extract semantically identical phrases from the same sentence. 

>In addition, Southwest allows each ticketed Customer to check one stroller and one car seat free of charge, in addition to the two free checked bags.

could be extracted to several phrases such as 

> allowing each ticketed Customer to check one stroller... 
> allows each ticketed Customer to check one stroller...

A key problem for both methods is actor identification. Both methods sometimes include activities performed by non-firm actors, including governments, suppliers, customers, products, or operating systems.

Another problem is that LLM may infer rather than extract, though this is occasional in the current stage of our exploration. For now, though LLM generally does a better job, I would recommend combining dependency parsing and LLM results for the next step. At this stage, we keep the medium version of dependency parsing and LLM, and combine them into a dataset of 910 phrases. The extraction process, especially dependency parsing, requires further tuning for more precise outputs. 

### Cleaning

 After extraction, the cleaning stage had four goals: normalize phrase text, filter obviously bad extractions, remove duplicate phrases, and identify the actor performing each activity.

First we lowercased phrases, lemmatized words, stripped generic stopwords and preserved strategically important words such as negation, mode, and scope terms. Then we removed obviously bad extractions, such as single-token phrases fragments and phrases with too little activity content. This reduced the data from 910 input rows to 899 retained rows.

We then conducted same-sentence deduplication. Within each firm-year-source-sentence group, we compared phrases that referred to the same underlying activity and retained the best representative phrase. This step mainly corrected extraction-level redundancy caused by different sentence parsing or chunk extraction patterns, reducing the dataset from 899 to 794 rows.

Because same-sentence deduplication only addresses local extraction duplicates, we then used DeepSeek for two additional cleaning tasks. First, DeepSeek labeled candidate phrase pairs as `duplicate`, `related_but_distinct`, or `not_duplicate`, allowing us to remove broader semantic duplicates. Second, DeepSeek identified the actor performing each activity, so that we could keep only activities performed by the focal firm and remove activities performed by customers, suppliers, governments, products, or other non-firm actors. After semantic deduplication and actor filtering, the final firm-only activity dataset contained 522 phrases.

However, the dataset after DeepSeek's cleaning is still not a perfect set of firm level activity phrases. We can try to manually label a subset of phrases and use it to evaluate or refine the LLM-based activity and actor identification procedure.


### Embedding

Using `sentence-transformers/all-mpnet-base-v2`, we embedded the 522 final phrases and checked near-duplicates across different cosine similarity thresholds.

|threshold|near-duplicate phrases|unique phrases|near-duplicate groups|near-duplicate pairs|
|---|---|---|---|---|
|0.88|48|474|24|25|
|0.89|40|482|20|21|
|0.90|30|492|15|15|
|0.91|26|496|13|13|
|0.92|20|502|10|10|
|0.93|18|504|9|9|
|0.94|14|508|7|7|
|0.95|12|510|6|6|
Even when the cosine similarity thresholds are already high, we still find unreasonable results. For example, when the threshold is 0.95:

>entered collective bargaining agreement ...

and 

>will negotiate collective bargaining agreement ...

are put into the same group, despite the very difference meaning of "enter" and "negotiate". Therefore, we do not collapse phrases based only on embedding similarity at this stage.

## Clustering, Stability and Sanity

Following the notes, I tried different clustering methods including K-means, HDBSCAN, Ward linkage and BERTopic. 

Based on the results, BERTopic offers the best interpretability. The clusters offer categories in Porter's value chain theory, including "sale, service center" and "campaign, advertising". 

Ward and K-means with k=50 offers good results as well, but the explainotory power is slightly lower than BERTopic. The clusters are not specified enough with smaller k, and will be fragmented when k=100 or 200. HDBSCAN performs badly in this case. Under some parameters the phrases are largely recognized as noise, and under other parameters almost all phrases are put into one cluster.

We measure stability using adjusted Rand index (ARI), which compares whether two clustering runs assign the same pairs of phrases to the same clusters.
We then evaluated clustering stability using three perturbation tests. First, because Tesla contributes 224 of the final 522 phrases, we excluded Tesla and reclustered the remaining firms. Second, we reran seed-sensitive methods with different random seeds. Third, we drew 80% subsamples and reclustered each subsample. We measure stability using adjusted Rand index (ARI), which compares whether two clustering runs assign the same pairs of phrases to the same clusters. In the table below, `n_tests` reports the number of ARI comparisons summarized in each row.

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

Although BERTopic is highly interpretable, its stability score is relatively low, and k-means is also unstable. This is expected because BERTopic depends on a multi-stage UMAP-HDBSCAN pipeline, while k-means depends on random centroid initialization. The small sample size and many small clusters further amplify the effects of changing seeds or corpus composition.

Finally we conduct a sanity check using the BERTopic outcome. Since it provides the most interpretable taxonomy. 51 clusters and a noise cluster are formed. We found some of the terms in clusters directly match with our knowledge in the activities of the firms, such as Tesla's "energy storage" and "gigafactory", Ford's "dealer" and "loan", Southwest's "reservation" and Delta's "alliance" and "global airlines". We also found industry spanning activities such as "extended service" and "service center" while others may be less specified phrases such as "agreement" and "bargaining". This may indicate the problems we had in the extraction stage. Ward, despite its stability, is less interpretable. For example, when it does not form a cluster for "advertising campaign" and "customer service" related activities, but mixed with other phrases.

Because the sample is small and diverse, many clusters are industry-specific or firm-specific, and Tesla contributes many distinctive technical terms. I expect that adding more industries and more firms within each industry would produce more industry-spanning activity categories.

The final clustering outputs were transformed into firm-activity matrices as requested in Step 6. The column names, which correspond to activity clusters, should be updated once we finalize human-readable cluster labels.

## Key Outputs

- Final activity phrases: `phrases/clean_normalize/actor_identification/activities_medium_actor_firm_only_deduplicated.csv`
- Final embeddings: `embedding/firm_only_deduplicated_mpnet_index.csv` and `embedding/firm_only_deduplicated_mpnet_embeddings.npy`
- Clustering results: `clustering/`
- Main BERTopic taxonomy: `clustering/bertopic/default_min_topic_size_05/`
- Main Ward comparison: `clustering/hierarchical_ward/k_050/`
- ARI summary: `stability/ari_summary_by_test_method.csv`
- Firm-activity matrices: `firm_act_matrix/`

## To Do

1. Filter the loose phrases to see if we missed any valuable phrases.
2. Manually label a set of extraction
3. Decide names for each cluster (maybe based on key terms?)



