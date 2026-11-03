import copy
import math
import unittest

from research_agent.controlled_evaluation import evaluate, normalize_qrels
from research_agent.reranker_experiment import digest

IDS = [f"C{i:02d}" for i in range(1, 11)]
CASE = {"query_id": "Q01", "query": "Comparison", "target_type": "paper", "criteria": "Useful evidence",
        "candidates": [{"candidate_id": cid, "node_id": f"node-{cid}", "type": "paper", "text": cid}
                       for cid in IDS]}
PACKAGE = {"schema": "reranker-pilot-v1", "run_id": "controlled", "cases": [CASE]}
GRADES = {"C01": 2, "C02": 2, "C03": 1, "C04": 1}
QRELS = {"schema": "reranker-qrels-v1", "cases": [{"query_id": "Q01", "labels": [
    {"candidate_id": cid, "grade": GRADES.get(cid, 0),
     "negative_kind": "hard" if cid in ("C05", "C06") else ("easy" if cid not in GRADES else None),
     "hard_negative": cid in ("C05", "C06")}
    for cid in IDS]}]}
RANKING = ["C05", "C01", "C03", "C07", "C08", "C02", "C04", "C06", "C09", "C10"]


def result(ranking=RANKING, accepted=None, kind="warm_or_cold_client_reuse", e2e=2.0):
    record = {"query_id": "Q01", "case_hash": digest(CASE),
              "ranking": [{"candidate_id": cid, "score": 0.5} for cid in ranking],
              "timing": {"kind": kind, "e2e_s": e2e}, "cache_hit": kind == "cache_replay"}
    if accepted is not None:
        record["accepted"] = accepted
    return {"run_id": "controlled", "client_init_s": 0.3, "results": [record]}


class ControlledEvaluationTests(unittest.TestCase):
    def test_constructed_ranking_exact_metrics(self):
        report = evaluate(PACKAGE, QRELS, {"jev": result(accepted=["C05", "C01"])})
        metrics = report["providers"]["jev"]["per_query"]["Q01"]
        ideal = 3 + 3 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
        observed5 = 3 / math.log2(3) + 1 / math.log2(4)
        observed10 = observed5 + 3 / math.log2(7) + 1 / math.log2(8)
        self.assertAlmostEqual(metrics["ndcg_at_5"], observed5 / ideal)
        self.assertAlmostEqual(metrics["ndcg_at_10"], observed10 / ideal)
        self.assertEqual(metrics["strong_recall_at_5"], 0.5)
        self.assertEqual(metrics["hard_negative_intrusion_count_at_5"], 1)
        self.assertEqual(metrics["hard_negative_intrusion_fraction_at_5"], 0.2)
        self.assertEqual((metrics["accepted_precision"], metrics["accepted_recall"],
                          metrics["accepted_retention"]), (0.5, 0.25, 0.2))
        self.assertEqual(report["providers"]["jev"]["macro"]["ndcg_at_5"], metrics["ndcg_at_5"])
        self.assertEqual(report["providers"]["jev"]["macro_metric_sample_n"]["ndcg_at_5"], 1)
        latency = report["providers"]["jev"]["latency"]
        self.assertEqual((latency["fresh_e2e_sample_n"], latency["fresh_e2e_median_s"],
                          latency["fresh_e2e_p95_s"], latency["client_init_s"]), (1, 2.0, 2.0, 0.3))

    def test_ideal_reversal_and_empty_acceptance(self):
        ideal = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10"]
        report = evaluate(PACKAGE, QRELS, {"voyage": result(ideal), "jev": result(list(reversed(ideal)), [])})
        self.assertEqual(report["providers"]["voyage"]["per_query"]["Q01"]["ndcg_at_5"], 1)
        reverse = report["providers"]["jev"]["per_query"]["Q01"]
        self.assertLess(reverse["ndcg_at_5"], 1)
        self.assertIsNone(reverse["accepted_precision"])
        self.assertEqual((reverse["accepted_recall"], reverse["accepted_retention"]), (0, 0))
        self.assertEqual(report["providers"]["jev"]["macro_metric_sample_n"]["accepted_precision"], 0)
        self.assertEqual(report["providers"]["jev"]["macro_metric_sample_n"]["accepted_retention"], 1)

    def test_alignment_and_qrels_fail_closed(self):
        malformed = copy.deepcopy(QRELS)
        malformed["cases"][0]["labels"].pop()
        with self.assertRaisesRegex(ValueError, "candidate IDs differ"):
            evaluate(PACKAGE, malformed, {"voyage": result()})
        duplicate = copy.deepcopy(QRELS)
        duplicate["cases"][0]["labels"].append(duplicate["cases"][0]["labels"][0])
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            normalize_qrels(duplicate, PACKAGE)
        bad_rank = result()
        bad_rank["results"][0]["ranking"][1]["candidate_id"] = "C05"
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            evaluate(PACKAGE, QRELS, {"voyage": bad_rank})
        bad_run = result()
        bad_run["run_id"] = "different"
        with self.assertRaisesRegex(ValueError, "run_id"):
            evaluate(PACKAGE, QRELS, {"voyage": bad_run})
        bad_hash = result()
        bad_hash["results"][0]["case_hash"] = "stale"
        with self.assertRaisesRegex(ValueError, "case hash"):
            evaluate(PACKAGE, QRELS, {"voyage": bad_hash})
        bad_kind = copy.deepcopy(QRELS)
        bad_kind["cases"][0]["labels"][0]["negative_kind"] = "hard"
        with self.assertRaisesRegex(ValueError, "negative_kind"):
            evaluate(PACKAGE, bad_kind, {"voyage": result()})
        bad_qrels_run = copy.deepcopy(QRELS)
        bad_qrels_run["run_id"] = "another-run"
        with self.assertRaisesRegex(ValueError, "qrels run_id"):
            evaluate(PACKAGE, bad_qrels_run, {"voyage": result()})

    def test_cache_and_external_flow_separation(self):
        replay = result(kind="cache_replay", e2e=0.001)
        assistant = result(kind="external_flow", e2e=30)
        report = evaluate(PACKAGE, QRELS, {"voyage": replay, "assistant": assistant})
        voyage = report["providers"]["voyage"]["latency"]
        self.assertEqual((voyage["fresh_e2e_sample_n"], voyage["cache_replay_sample_n"]), (0, 1))
        self.assertIsNone(voyage["fresh_e2e_median_s"])
        external = report["providers"]["assistant"]["latency"]
        self.assertEqual((external["external_flow_sample_n"], external["external_flow_median_s"]), (1, 30))
        self.assertEqual(external["fresh_e2e_sample_n"], 0)

    def test_two_disjoint_curator_files_and_p95(self):
        package = copy.deepcopy(PACKAGE)
        second = copy.deepcopy(CASE)
        second["query_id"] = "Q02"
        package["cases"].append(second)
        first_qrels = copy.deepcopy(QRELS)
        second_qrels = copy.deepcopy(QRELS)
        second_qrels["cases"][0]["query_id"] = "Q02"
        results = result()
        second_result = copy.deepcopy(results["results"][0])
        second_result["query_id"] = "Q02"
        second_result["case_hash"] = digest(second)
        second_result["timing"]["e2e_s"] = 4.0
        results["results"].append(second_result)
        report = evaluate(package, [first_qrels, second_qrels], {"voyage": results})
        latency = report["providers"]["voyage"]["latency"]
        self.assertEqual(latency["fresh_e2e_sample_n"], 2)
        self.assertEqual(latency["fresh_e2e_median_s"], 3.0)
        self.assertAlmostEqual(latency["fresh_e2e_p95_s"], 3.9)


if __name__ == "__main__":
    unittest.main()
