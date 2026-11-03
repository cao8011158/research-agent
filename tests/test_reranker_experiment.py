import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from research_agent.reranker_experiment import (
    digest, import_assistant, jev_payload, neutral_packet, run_jev, run_package, run_voyage,
)

CASE = {"query_id": "Q01", "query": "Compare A and B", "target_type": "paper",
        "criteria": "Useful evidence about A or B", "candidates": [
            {"candidate_id": "C01", "node_id": "secret-node-1", "type": "paper", "text": "A has a latency limit."},
            {"candidate_id": "C02", "node_id": "secret-node-2", "type": "paper", "text": "B has a safety limit."}]}


class FakeJev:
    model, timeout, max_retries = "jev-1.14.0", 3, 2
    last_request_audit = {"attempts": 2, "queue_duration_s": 0.01}

    def __init__(self, answers=None, delay=0):
        self.answers, self.delay, self.calls = answers, delay, 0

    def decide(self, state, questions):
        self.calls += 1
        self.state, self.questions = state, questions
        time.sleep(self.delay)
        answers = self.answers if self.answers is not None else {
            "C01": {"type": "choice", "choice": "relevant", "confidence": 0.91,
                    "probabilities": {"relevant": 0.69, "not_relevant": 0.30, "insufficient": 0.01}},
            "C02": {"type": "choice", "choice": "not_relevant", "confidence": 0.99,
                    "probabilities": {"relevant": 0.71, "not_relevant": 0.28, "insufficient": 0.01}}}
        return {"model": self.model, "answers": answers, "usage": {"total_tokens": 17}, "attempts": 2}


class FakeVoyage:
    def __init__(self, rows=None, delay=0):
        self.rows, self.delay, self.calls = rows, delay, 0

    def rerank(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        time.sleep(self.delay)
        rows = self.rows if self.rows is not None else [(1, 0.9), (0, 0.2)]
        return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in rows],
                               model="rerank-3", total_tokens=23)


class PilotTests(unittest.TestCase):
    def test_neutral_full_jev_state(self):
        packet, body = neutral_packet(CASE), jev_payload(CASE)
        self.assertNotIn("secret-node", json.dumps(packet))
        self.assertNotIn("secret-node", json.dumps(body))
        self.assertEqual(len(body["state"]["candidates"]), 2)
        self.assertEqual(set(body["questions"]), {"C01", "C02"})
        self.assertIn("state.candidates[0]", body["questions"]["C01"]["instructions"])
        self.assertIn("A has a latency limit", body["questions"]["C01"]["instructions"])
        self.assertEqual(set(body["questions"]["C01"]["criteria"]),
                         {"relevant", "not_relevant", "insufficient"})

    def test_jev_probability_choice_no_padding_timing(self):
        client = FakeJev(delay=0.01)
        result = run_jev(CASE, client)
        self.assertEqual(result["accepted"], [])
        self.assertEqual([r["candidate_id"] for r in result["ranking"]], ["C02", "C01"])
        self.assertEqual(result["ranking"][0]["confidence"], 0.99)
        self.assertEqual(result["attempts"], 2)
        self.assertGreaterEqual(result["timing"]["e2e_s"], result["timing"]["provider_call_wall_s"])
        self.assertGreaterEqual(result["timing"]["provider_call_wall_s"], 0.01)
        self.assertEqual(len(client.state["candidates"]), 2)

    def test_jev_malformed_preflight(self):
        bad = FakeJev(answers={"C01": {"type": "choice", "choice": "relevant",
                                       "probabilities": {"relevant": 1.0}}})
        with self.assertRaisesRegex(ValueError, "cover every"):
            run_jev(CASE, bad)
        oversized = json.loads(json.dumps(CASE))
        oversized["candidates"][0]["text"] = "x" * 41000
        with self.assertRaisesRegex(ValueError, "Full-pool"):
            run_jev(oversized, bad)
        self.assertEqual(bad.calls, 1)

    def test_voyage_rejects_bad_results(self):
        for rows in ([(0, 0.5), (0, 0.4)], [(0, float("nan")), (1, 0.2)], [(0, 0.4)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                run_voyage(CASE, FakeVoyage(rows))

    def test_voyage_payload_cache(self):
        package = {"schema": "reranker-pilot-v1", "run_id": "run", "cases": [CASE]}
        client = FakeVoyage(delay=0.01)
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            first = run_package(package, "voyage", directory, client=client)
            second = run_package(package, "voyage", directory, client=client)
            self.assertEqual((first["fresh_count"], second["cache_hit_count"], client.calls), (1, 1, 1))
            self.assertEqual(client.kwargs, {"model": "rerank-3", "query": CASE["query"] + "\nExplicit relevance criteria: " + CASE["criteria"],
                                           "documents": [c["text"] for c in CASE["candidates"]],
                                           "top_k": 2, "truncation": False})
            self.assertGreaterEqual(first["results"][0]["timing"]["e2e_s"], 0.01)
            self.assertEqual(second["results"][0]["timing"]["kind"], "cache_replay")
            self.assertEqual(second["results"][0]["original_timing"], first["results"][0]["timing"])

    def test_assistant_permutation_timing(self):
        response = {"query_id": "Q01", "ranking": [{"candidate_id": "C02", "score": 0.8},
                                                     {"candidate_id": "C01", "label": "insufficient"}],
                    "started_at": "2026-11-03T01:00:00Z", "finished_at": "2026-11-03T01:00:12Z"}
        result = import_assistant(CASE, response)
        self.assertEqual(result["timing"]["e2e_s"], 12)
        self.assertIsNone(result["timing"]["model_processing_s"])
        self.assertEqual(result["score_semantics"], "self_reported_not_calibrated")
        response["ranking"][1]["candidate_id"] = "C02"
        with self.assertRaisesRegex(ValueError, "permutation"):
            import_assistant(CASE, response)

    def test_hash_full_case(self):
        changed = json.loads(json.dumps(CASE))
        changed["candidates"][0]["node_id"] = "another-node"
        self.assertNotEqual(digest(changed), digest(CASE))


if __name__ == "__main__":
    unittest.main()
