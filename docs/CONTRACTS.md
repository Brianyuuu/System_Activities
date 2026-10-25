# Data contracts and methods

## Source verification

`scripts/prepare_extraction.py` consumes one intended primary annual filing per economic entity/fiscal year. Required CSV fields are:

```
company_id,ticker,fiscal_year,cik,accession,form,report_period_end,filing_date,
company_name_at_year,registrant_names,identity_basis,fiscal_year_basis,
verification_scope,included_in_has10k,panel_has10k,industry
```

Use canonical economic-entity IDs in `company_id` (letters, numbers, underscore or hyphen), not reused ticker symbols. `accession` uses `##########-##-######`; dates use ISO dates. `registrant_names` can contain historical aliases separated by ` | `. Identity and fiscal-year bases must describe evidence, not just an assertion that names look similar. Both scope flags must be `1`. Upstream historical identity resolution is an explicit research input, not an automated entity-resolution claim.

Provide `local_raw_path` and `raw_sha256` for a local original filing; relative paths are resolved against the mapping CSV. Otherwise supply `sec_primary_document_url` and/or `sec_full_submission_url` pointing to official SEC archive documents for that accession. Set `SEC_USER_AGENT` for downloads. The local mapping hash verifies the supplied original, not its SEC authenticity; retain the official document URL and acquisition evidence.

Preparation verifies the local raw hash, document CIK when present, historical name, fiscal year, period, primary-document selection, and Item 1 boundary/extent. Mapping conflicts and ambiguous multiple filings remain held. One accession cannot be assigned to multiple company-years. Non-calendar fiscal-year labels require explicit evidence. The selected text is **primary Item 1 only**: references incorporated from elsewhere are not fetched. Review `source/audits/source_packets/`, `outputs/coverage_inventory.csv`, and `audits/item1_only_build.json` before execution. Resolve held records by correcting the input/evidence and preparing a fresh release; do not silently mark a held packet approved.

## Extraction

The deployed orchestration in `runtime/lib/latest_phrase_pipeline.py` performs clause/context extraction, focal-company filtering, mechanism-type audit, decomposition, atomization, contextual audit, deduplication, entity replacement, specificity filtering, and final export. Strict runtime receipts record commands, inputs, outputs, attempts, and hashes. Failure-to-source/reject/parent fallback switches are removed; transport failures do not become classifications. Each filing uses eight API workers in this release.

A complete report requires technical QA, input/output hashes, source-sentence provenance, no unresolved extraction jobs or transport-fallback rows, and complete strict stages. Phrase-level semantic flags are retained. `semantic_release_pass=false` is deliberate: technical validity is not individual semantic certification.

Export one exact source industry at a time with `export_phrases.py --industry "SOURCE NAME"`. Canonical `company_id` becomes assignment `ticker`; the old ticker is retained as `source_ticker`. All enriched fields remain, and `final_phrase` is copied from `atom_mechanism_phrase`. Approved reports must all be complete for that export. Held years remain `complete=0` coverage rows; completed zero-phrase reports remain `complete=1, expected_source_rows=0`. Review any entirely empty industry separately: the assignment adapter requires at least one phrase, and does not fabricate an empty completed run.

## Assignment and review

Input CSVs require a nonempty phrase column, `original_sentence`, `year`, and `ticker` or `firm_id`. Exact original columns are conserved. Whitespace-normalized phrases share a GLM unit only when their relevant sentence context agrees. The word-count regex counts alphanumeric words with internal hyphens/apostrophes. GLM retains context at <=4 words; Gemini at <4 words. Long phrases omit sentence context.

Taxonomy columns: unique `parent_id`, nonempty `parent_label`, optional `inclusion_rule`, optional `exclusion`. The adapter strips surrounding whitespace, falls back to label for missing scope, appends exclusions to scope, and also retains the separate exclusion field. The bundled projection preserves all consumed fields and row order.

Retrieval uses `text-embedding-3-small`, 1,536 dimensions, unit vectors, and the maximum similarity across phrase/alias/context views versus parent label/scope views. New default policy is global top 1,000 intersect score >0.33, with ties broken by ascending frozen parent index. The historical uncapped >=0.33 policy is selectable and frozen per run. GLM high-recall screening uses `parcc/GLM`, temperature 0 and disabled thinking; unresolved/deferred pairs block completion.

Gemini candidates are GLM-kept pairs union mandatory top 30 across the full frozen taxonomy, including candidates outside the recall threshold when needed. `gemini-3.1-flash-lite` batch review first groups candidates by parent; only parent-positive pairs receive reciprocal phrase-first review. Acceptance requires both positive. Bounded batch repairs address malformed/missing responses; unresolved judgments never become forced negatives. Gemini uses parent labels; its prompts are frozen copies under `pipeline/assignment/prompts/`.

`outputs/pipeline/final/all_assigned.csv` retains every input row, including unmatched phrases. `source_parent_assignments.csv` is the expanded accepted edge list. Final `validation.json` must have `complete=true`, `unresolved=0`, and matching source/taxonomy/output hashes. Freeze inputs and code per run; concurrency changes do not change the scientific configuration.

## Matrix inputs

Coverage CSV fields: `firm,year,complete,expected_source_rows`. Every assignment row must belong to coverage, and row counts must agree exactly. Add missing target firm-years explicitly with `complete=0`; an omitted row cannot be inferred as a missing year. A completed year with no phrases can have `complete=1, expected_source_rows=0`.

World CSV fields: `firm,year,parent_id,parent_label,source_id`. The JSON sidecar requires `taxonomy_sha256`, `world_sha256`, `basis` (`confirmed_assignments` or `current_catalog_memberships`), and a nonempty `scope_description`. `build_world.py` constructs the confirmed-assignment form from one or more complete runs using the same frozen taxonomy. Existing catalog memberships require an independently reviewed export with source firm/year and those same fields. Do not manufacture memberships from taxonomy labels alone.

World first-observation years are calculated across that declared corpus plus complete target assignments. Target firm-years marked missing are excluded even if they occur in World input. “New to World” counts parents whose earliest available year equals the current year. Different corpus scopes change this metric; preserve the sidecar with the workbook.

## Matrix arithmetic and presentation

Each industry's workbook has one firm sheet plus `firm year panel`. Firm sheets contain exact parent labels as rows and declared years as columns; presence is binary. Missing-year zeros have comments explaining they are placeholders. Only presence cells with value 1 are light green. Fonts are Arial 10; freezes are B2 for firm sheets and C2 for the panel. Firm IDs must be valid unique Excel sheet names of at most 31 characters.

The panel columns are firm, year, parents count, parents enter, parents exit, parents new to world, closest firm, distance to closest firm, distance to avg. Enter/exit compare only adjacent complete calendar years; gaps are blank, not zero. Missing years have blank metrics. Labels are exact for binary counts; NFKC + casefold + collapsed whitespace applies only to semantic vectors.

Semantic comparison uses `gemini-embedding-2`, 3,072 dimensions, with text prefix `task: sentence similarity | query: `. Normalize each embedding, then subtract the equal-weight mean of unique active normalized labels in this industry. **Do not normalize the centered parent vectors again.** Let N be complete firm-years and df the number containing a normalized label. Weight each centered parent vector by `sqrt(1 + ln((N+1)/(df+1)))`, sum within a firm-year, then unit-normalize the sum. Empty or numerically zero sums have no semantic distances.

Compare only other firms in the same industry/year with valid vectors. Distance is `1 - cosine`; “distance to avg” is the mean of pairwise distances, not distance to a centroid. Ties select the alphabetically first firm. The builder checks the cosine/squared-Euclidean identity, serializes the workbook, and reads it back to check cells. It does not claim a visual rendering audit; `visual_render_validation=false` remains explicit. This portable builder implements the documented method; it is not a byte-identical copy of old path-specific workbooks.
