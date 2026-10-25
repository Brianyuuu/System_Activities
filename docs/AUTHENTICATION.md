# Authentication, credentials and request flow

This repository is a research batch client, not a web application. It has no application user sessions, login routes, JWT issuer, refresh-token service or database-backed account system. Authorization comes from the operator's PARCC/Slurm access and API credentials issued by each provider.

## Components

| Client | Credential source | Outbound use |
|---|---|---|
| Original exploration in `Empirical_explore/codes/` | Historical DeepSeek environment-key setup; see original README | Original exploratory model calls; separate from current pipeline |
| SEC source acquisition, `prepare_from_mapping.py` | `SEC_USER_AGENT` identifies the requester | HTTPS official SEC archive GET requests; no bearer token |
| Extraction, `runtime/lib/latest_phrase_pipeline.py` | `LITELLM_API_KEY` process environment | OpenAI-compatible PARCC GLM client; API credential sent to configured LiteLLM endpoint |
| Assignment, `run_glm_only.py` / `pipeline_common.py` | External mode-0600 JSON via `PIPELINE_SECRET_FILE` | `OSS_API_KEY` as Bearer credential to PARCC; `OPENAI_API_KEY` as Bearer credential to OpenAI embeddings |
| Gemini reciprocal review, `run_reciprocal.py` / `gemini_parent_first_batch.py` | Same external JSON's `GEMINI_API_KEY`, copied to process environment | `x-goog-api-key` header for upload, batch creation, polling and output retrieval |
| Matrix embedding, `scripts/matrix.py` | `GEMINI_API_KEY` environment | `x-goog-api-key` header to Gemini `batchEmbedContents` |

`OSS_API_KEY` is the assignment package's historical name for the PARCC GLM API key. It is not an object-storage credential. Extraction calls the same service with the name `LITELLM_API_KEY`; configure both from your authorized PARCC credential where appropriate.

## Request flow and persistence

The operator authenticates to PARCC independently of this code, deploys the repository and starts a job. Local preparation reads no model credentials. Before a live assignment stage, the launcher checks frozen inputs, reads the external secret file, verifies that group/other permission bits are absent, and copies only the required values to its process environment. The HTTP helper constructs Authorization headers. The extraction wrapper requires its key in the environment; its inherited dotenv loader points to a nonbundled `environment-only.env` fallback.

Gemini uploads frozen request JSONL, records submission intent, submits the batch, persists its resource name, polls it, downloads responses, and validates the response schema and row coverage. Batch job names, upload resources, hashes and ledger IDs are resumability metadata, not application access tokens. Uncertain submissions remain visible for reconciliation. Model token counts in ledgers mean billed input/output units; they are distinct from authentication credentials.

Successful response caches, API error excerpts, logs, request payloads, usage records and SQLite reservations may contain research text. They belong in ignored run storage, not commits. The code does not intentionally serialize credential values into those records. Avoid enabling verbose HTTP/header debugging or printing the environment. A key is neither embedded in this repository nor automatically refreshed; expired/revoked keys require replacement through the issuing provider and the protected runtime configuration.

PARCC SSH credentials, SSO/Kerberos tickets and Slurm allocation permissions remain managed by the cluster. The repository does not copy them, create users, or alter remote access controls. Changing a credential is an operational action; changing model inputs/prompts/recall policy changes the scientific run and requires a new frozen run.
