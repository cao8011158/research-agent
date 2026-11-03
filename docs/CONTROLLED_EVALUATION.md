# Controlled reranker evaluation

This utility scores **only** frozen reranker outputs against independently prepared labels. It makes no provider calls, retrieves no candidates, and fits no threshold. All controlled cases are evaluation cases; Jev's `0.70` acceptance rule is inherited from the frozen pilot protocol.

## Inputs

- `--cases`: frozen `reranker-pilot-v1` package.
- `--qrels`: one or more disjoint `reranker-qrels-v1` files. Each case has `query_id` and `labels`, each label has `candidate_id`, integer `grade` in 0/1/2, and `negative_kind` (`hard` or `easy` for grade 0; `null` for grades 1/2). An optional `hard_negative` Boolean must agree with `negative_kind`. Curator explanation fields are ignored by metrics.
- `--result`: repeat `voyage=PATH`, `jev=PATH`, or `assistant=PATH`. These are the JSON summaries produced by the reranker driver and assistant import. Each must contain the exact run ID, all query IDs, each result's full-case hash, and a complete unique candidate permutation.

Example:

```text
python scripts/evaluate_controlled_rerank.py --cases cases.json --qrels qrels_a.json --qrels qrels_b.json --result voyage=voyage_results.json --result jev=jev_results.json --result assistant=assistant_results.json --output controlled_metrics.json
```

If the curators supply one combined qrels file, use one `--qrels`. Any missing, extra, or duplicate query, candidate, label, or ranking fails before a report is written. Qrels should remain sealed until all provider rankings are frozen.

## Fixed metrics

- `nDCG@5` and `nDCG@10`: gains `2^grade - 1`, discount `log2(rank+1)` with rank starting at 1, ideal ordering over the full labeled pool. If no relevant item exists, nDCG is null.
- `strong_recall_at_5`: grade-2 items in Top 5 divided by all grade-2 items. Null if there is no grade-2 item.
- Hard-negative intrusion: number of grade-0 `negative_kind=hard` items in Top 5 and that number divided by **5**.
- Jev acceptance: `grade > 0` precision and recall among its recorded accepted IDs, plus retention `accepted_count/pool_size`. Empty acceptance gives null precision, zero recall when positives exist, and zero retention. No assistant score threshold is inferred.

The report includes each query, a macro mean of defined query metrics, and `macro_metric_sample_n` for every metric. Null metrics are omitted from the macro denominator rather than turned into zero; empty acceptance still contributes zero retention. A qrels `run_id`, when present, must match the case package. Latency has separate fresh end-to-end samples, median, linearly interpolated p95, sample count, client initialization, cache replay samples, and assistant external-flow samples. The assistant flow includes handoff and human/tool orchestration; its model inference component is unknown and is not directly comparable with provider call time.

Jev's `P(relevant)` estimates the Choice label for useful relevance (`grade > 0` in this evaluation). It is not expected graded contribution strength. nDCG distinguishes grades 2 and 1, while acceptance precision, recall, and retention show the effect of Jev's fixed filter. Read these metrics together; no new model or protocol variant is introduced here.

Offline test: `python -m unittest discover -s tests -p test_controlled_evaluation.py -v` with this repository's `src` on `PYTHONPATH`.
