"""Preregistered quality and latency metrics for frozen reranker rankings."""
from __future__ import annotations

import math
import statistics

from research_agent.reranker_experiment import digest, validate_package

QRELS_SCHEMA = "reranker-qrels-v1"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mean(values):
    return statistics.mean(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _map_unique(items, key, context):
    if not isinstance(items, list):
        raise ValueError(f"{context} must be a list")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(key), str) or not item[key]:
            raise ValueError(f"{context} requires nonempty {key}")
        if item[key] in result:
            raise ValueError(f"duplicate {key} in {context}: {item[key]}")
        result[item[key]] = item
    return result


def normalize_qrels(documents, package):
    """Accept one or several disjoint curator files; require complete coverage."""
    validate_package(package)
    if isinstance(documents, dict):
        documents = [documents]
    if not isinstance(documents, list) or not documents:
        raise ValueError("qrels document required")
    merged = {}
    for document in documents:
        if not isinstance(document, dict) or document.get("schema") != QRELS_SCHEMA:
            raise ValueError(f"qrels schema must be {QRELS_SCHEMA}")
        if "run_id" in document and document["run_id"] != package["run_id"]:
            raise ValueError("qrels run_id differs from case package")
        cases = _map_unique(document.get("cases"), "query_id", "qrels cases")
        for query_id, case in cases.items():
            if query_id in merged:
                raise ValueError(f"duplicate qrels query_id across files: {query_id}")
            merged[query_id] = case
    input_cases = _map_unique(package["cases"], "query_id", "input cases")
    if set(merged) != set(input_cases):
        raise ValueError(f"qrels query IDs differ: missing={sorted(set(input_cases)-set(merged))}, extra={sorted(set(merged)-set(input_cases))}")
    labels_by_case = {}
    for query_id, case in merged.items():
        labels = _map_unique(case.get("labels"), "candidate_id", f"qrels {query_id}")
        candidates = {c["candidate_id"] for c in input_cases[query_id]["candidates"]}
        if set(labels) != candidates:
            raise ValueError(f"qrels candidate IDs differ for {query_id}: missing={sorted(candidates-set(labels))}, extra={sorted(set(labels)-candidates)}")
        normalized = {}
        for cid, label in labels.items():
            grade = label.get("grade")
            kind = label.get("negative_kind")
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError(f"invalid grade for {query_id}/{cid}")
            if ("negative_kind" not in label or
                    (grade == 0 and kind not in ("hard", "easy")) or
                    (grade > 0 and kind is not None)):
                raise ValueError(f"invalid negative_kind for {query_id}/{cid}")
            hard = kind == "hard"
            if "hard_negative" in label and (type(label["hard_negative"]) is not bool or
                                             label["hard_negative"] != hard):
                raise ValueError(f"hard_negative disagrees with negative_kind for {query_id}/{cid}")
            normalized[cid] = {"grade": grade, "hard_negative": hard}
        labels_by_case[query_id] = normalized
    return labels_by_case


def _dcg(grades, k):
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(grades[:k]))


def _quality(ranked_ids, labels, accepted=None):
    grades = [labels[cid]["grade"] for cid in ranked_ids]
    ideal = sorted((value["grade"] for value in labels.values()), reverse=True)
    metrics = {}
    for k in (5, 10):
        denominator = _dcg(ideal, k)
        metrics[f"ndcg_at_{k}"] = _dcg(grades, k) / denominator if denominator else None
    strong_total = sum(value["grade"] == 2 for value in labels.values())
    metrics["strong_recall_at_5"] = (
        sum(labels[cid]["grade"] == 2 for cid in ranked_ids[:5]) / strong_total
        if strong_total else None)
    intrusion = sum(labels[cid]["hard_negative"] for cid in ranked_ids[:5])
    metrics["hard_negative_intrusion_count_at_5"] = intrusion
    metrics["hard_negative_intrusion_fraction_at_5"] = intrusion / 5
    if accepted is not None:
        positive_total = sum(value["grade"] > 0 for value in labels.values())
        positive_accepted = sum(labels[cid]["grade"] > 0 for cid in accepted)
        metrics["accepted_precision"] = positive_accepted / len(accepted) if accepted else None
        metrics["accepted_recall"] = positive_accepted / positive_total if positive_total else None
        metrics["accepted_retention"] = len(accepted) / len(labels)
        metrics["accepted_count"] = len(accepted)
    return metrics


def _validate_results(package, result_document, provider_name):
    if not isinstance(result_document, dict):
        raise ValueError(f"{provider_name} result must be an object")
    if result_document.get("run_id") != package["run_id"]:
        raise ValueError(f"{provider_name} run_id differs")
    input_cases = _map_unique(package["cases"], "query_id", "input cases")
    results = _map_unique(result_document.get("results"), "query_id", f"{provider_name} results")
    if set(results) != set(input_cases):
        raise ValueError(f"{provider_name} query IDs differ")
    for query_id, result in results.items():
        candidates = {c["candidate_id"] for c in input_cases[query_id]["candidates"]}
        if result.get("case_hash") != digest(input_cases[query_id]):
            raise ValueError(f"{provider_name}/{query_id} case hash differs")
        ranking = result.get("ranking")
        if not isinstance(ranking, list) or len(ranking) != len(candidates):
            raise ValueError(f"{provider_name}/{query_id} ranking length differs")
        ranked = _map_unique(ranking, "candidate_id", f"{provider_name}/{query_id} ranking")
        if set(ranked) != candidates:
            raise ValueError(f"{provider_name}/{query_id} ranking IDs differ")
        for row in ranking:
            if "score" in row and not _number(row["score"]):
                raise ValueError(f"{provider_name}/{query_id} invalid score")
        if "accepted" in result:
            accepted = result["accepted"]
            if (not isinstance(accepted, list) or
                    any(not isinstance(cid, str) for cid in accepted) or
                    len(set(accepted)) != len(accepted) or not set(accepted) <= candidates):
                raise ValueError(f"{provider_name}/{query_id} invalid accepted IDs")
    return results


def _latency(results, provider_name, result_document):
    fresh, replay, external = [], [], []
    for query_id, result in results.items():
        timing = result.get("timing")
        if not isinstance(timing, dict) or not _number(timing.get("e2e_s")) or timing["e2e_s"] < 0:
            raise ValueError(f"{provider_name}/{query_id} missing finite nonnegative E2E timing")
        if result.get("cache_hit") is True or timing.get("kind") == "cache_replay":
            if result.get("cache_hit") is not True or timing.get("kind") != "cache_replay":
                raise ValueError(f"{provider_name}/{query_id} inconsistent replay timing")
            replay.append(timing["e2e_s"])
        elif provider_name == "assistant" or timing.get("kind") == "external_flow":
            external.append(timing["e2e_s"])
        else:
            fresh.append(timing["e2e_s"])
    init_s = result_document.get("client_init_s")
    if init_s is not None and (not _number(init_s) or init_s < 0):
        raise ValueError(f"{provider_name} invalid client_init_s")
    return {"fresh_e2e_sample_n": len(fresh), "fresh_e2e_samples_s": fresh,
            "fresh_e2e_median_s": statistics.median(fresh) if fresh else None,
            "fresh_e2e_p95_s": _percentile(fresh, 0.95),
            "cache_replay_sample_n": len(replay), "cache_replay_samples_s": replay,
            "external_flow_sample_n": len(external), "external_flow_samples_s": external,
            "external_flow_median_s": statistics.median(external) if external else None,
            "external_flow_p95_s": _percentile(external, 0.95),
            "client_init_s": init_s,
            "external_model_processing_s": None if external else None}


def evaluate(package, qrels_documents, provider_documents):
    """Compute fixed metrics; never fit thresholds or infer missing gold labels."""
    validate_package(package)
    labels_by_case = normalize_qrels(qrels_documents, package)
    if not isinstance(provider_documents, dict) or not provider_documents:
        raise ValueError("provider results required")
    output = {"schema": "reranker-controlled-evaluation-v1", "run_id": package["run_id"],
              "metric_spec": {"ndcg_gain": "2^grade-1", "strong_positive_grade": 2,
                              "positive_grade": ">0", "hard_negative": "negative_kind=hard",
                              "intrusion_fraction_denominator": 5,
                              "empty_accepted_precision": None}, "providers": {}}
    for provider_name, document in provider_documents.items():
        results = _validate_results(package, document, provider_name)
        per_query = {}
        for query_id in labels_by_case:
            result = results[query_id]
            ranked_ids = [row["candidate_id"] for row in result["ranking"]]
            accepted = result.get("accepted") if provider_name == "jev" else None
            if provider_name == "jev" and accepted is None:
                raise ValueError(f"Jev accepted list missing for {query_id}")
            per_query[query_id] = _quality(ranked_ids, labels_by_case[query_id], accepted)
        keys = list(next(iter(per_query.values())))
        samples = {key: [metrics[key] for metrics in per_query.values()
                         if metrics[key] is not None] for key in keys}
        macro = {key: _mean(values) for key, values in samples.items()}
        macro_sample_n = {key: len(values) for key, values in samples.items()}
        output["providers"][provider_name] = {
            "per_query": per_query, "macro": macro, "macro_metric_sample_n": macro_sample_n,
            "latency": _latency(results, provider_name, document)}
    return output
