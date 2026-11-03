# Local reranker pilot

This driver consumes a frozen candidate package. It does not retrieve candidates or produce gold judgments. Use the same package for each provider and keep the package and generated outputs outside published source control when they contain local KG text.

## Input

```json
{"schema":"reranker-pilot-v1","run_id":"pilot-20261103","cases":[{"query_id":"Q01","query":"Original query","target_type":"paper","criteria":"Explicit rule for useful evidence","candidates":[{"candidate_id":"C01","node_id":"local-only ID","type":"paper","text":"Frozen semantic candidate text"}]}]}
```

Each case has 1–20 candidates with unique neutral IDs. `node_id` stays local. Extra retrieval fields may be carried in the package but never enter semantic payloads. Whole-case hashes include them, so changing any frozen input invalidates the cache.

## Commands

Set `PYTHONPATH` to this repository's `src` plus `D:/Projects/kg-builder/src`. Use `D:/Projects/kg-builder/.venv/Scripts/python.exe` for the pilot:

```text
python scripts/run_reranker_experiment.py voyage --input CASES.json --output-dir OUT
python scripts/run_reranker_experiment.py jev --input CASES.json --output-dir OUT --threshold 0.70
python scripts/run_reranker_experiment.py export-assistant --input CASES.json --output-dir OUT
python scripts/run_reranker_experiment.py import-assistant --input CASES.json --assistant-results ASSISTANT.json --output-dir OUT
```

The Voyage adapter calls the installed official SDK with `model='rerank-3'`, `top_k=N`, `truncation=False`. Its query combines the original query and explicit relevance criteria. It initializes the SDK with a 120-second timeout and `max_retries=2`; the configuration is recorded, while actual SDK retry count is opaque. SDK 0.5.0's `RerankingObject` exposes `results` and `total_tokens`, but not a resolved model field. The Jev adapter uses the existing builder `JevClient` with one full-pool state and a separate three-class Choice per candidate. It checks the existing 40,000-character state and 60,000-byte body caps before calling. The complete state is never divided into partial candidate pools.

For Jev, all candidates receive a `P(relevant)` score. The unfiltered ranking sorts by that score then candidate ID. The accepted list requires **both** selected `relevant` and `P(relevant) >= 0.70`; empty results are valid. Jev `confidence` is recorded but never used as `P(relevant)`. The threshold is a prespecified pilot rule, not a calibrated project optimum.

Assistant export creates one neutral full-pool packet per case. Import expects a JSON list with one object per case:

```json
[{"query_id":"Q01","ranking":[{"candidate_id":"C01","score":0.8,"label":"relevant"}],"started_at":"2026-11-03T01:00:00Z","finished_at":"2026-11-03T01:00:12Z"}]
```

The ranking must include each candidate exactly once. `score` and `label` are optional; any score is self-reported and not treated as calibrated. Timestamps mark the root-controlled external flow from packet handoff to complete ranking; model processing time remains unknown. No Jev threshold is applied to assistant scores.

## Evidence and timing

Each result records the full case hash, semantic payload hash, exact semantic request, validated response, usage if present, ranking, timestamps, end-to-end rerank time, and provider-call wall time. Jev additionally records the client audit and known attempt count. End-to-end time begins with the supplied candidate case entering the adapter and includes serialization, preflight, all provider waits/retries, response validation, sorting and filtering. Retrieval preparation and client initialization are separate. The run summary records fresh samples, fresh median, cache hits, client initialization and whole-run wall time. A repeated exact case uses a cache file unless `--force` is set; replay timing is labeled `cache_replay`, with the original fresh timing preserved separately. Cache timing must not be reported as fresh latency.

Offline checks: `python -m unittest discover -s tests -p test_reranker_experiment.py -v`. This makes no provider calls.
