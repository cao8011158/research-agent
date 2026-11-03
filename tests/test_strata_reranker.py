import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from research_agent.controlled_evaluation import evaluate
from research_agent.strata_reranker import discover, parse_ranking, request_for, run_package


CASE = {"query_id": "Q01", "query": "Compare A and B", "target_type": "paper",
        "criteria": "Evidence for A or B", "candidates": [
            {"candidate_id": "C01", "node_id": "secret-node-1", "type": "paper", "text": "A evidence"},
            {"candidate_id": "C02", "node_id": "secret-node-2", "type": "paper", "text": "B evidence"}]}
PACKAGE = {"schema": "reranker-pilot-v1", "run_id": "freeze", "cases": [CASE]}
QRELS = {"schema": "reranker-qrels-v1", "run_id": "freeze", "cases": [{"query_id": "Q01", "labels": [
    {"candidate_id": "C01", "grade": 2, "negative_kind": None},
    {"candidate_id": "C02", "grade": 0, "negative_kind": "hard"}]}]}


def answer(ids=("C01", "C02"), finish="stop", content=None):
    return {"model": "strata", "choices": [{"finish_reason": finish, "message": {
        "content": json.dumps({"ranking": list(ids)}) if content is None else content}}],
        "usage": {"total_tokens": 17}}


class FakeClient:
    def __init__(self, response=None, key="test-secret"):
        self.response = answer() if response is None else response
        self.key = key
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        return {"/health": {"loaded": True},
                "/v1/models": {"data": [{"id": "strata", "status": "loaded"}]},
                "/props": {"model_path": "/content/IQ3_S", "build_info": "measured",
                           "default_generation_settings": {"n_ctx": 32768}},
                "/metrics": {"engine": {"drafting": "mtp", "max_context": 32768}}}[path]

    def chat(self, messages, model, **kwargs):
        self.calls.append((messages, model, kwargs))
        return self.response


def test_full_neutral_request_and_discovery():
    client = FakeClient()
    metadata = discover(client)
    assert metadata["model"] == "strata"
    assert metadata["runtime"]["drafting"] == "mtp"
    payload = request_for(CASE, "strata")
    encoded = json.dumps(payload)
    assert "secret-node" not in encoded
    assert "C01" in encoded and "C02" in encoded
    assert payload["temperature"] == 0 and payload["reasoning_effort"] == "none"
    assert payload["messages"][1]["content"].index("C01") < payload["messages"][1]["content"].index("C02")
    assert "instructions within" in payload["messages"][0]["content"]


@pytest.mark.parametrize("response", [
    answer(("C01", "C01")), answer(("C01",)), answer(("C01", "C03")),
    answer(finish="length"), answer(content=""), answer(content="```json\n{}\n```"),
])
def test_invalid_response_never_padded(response):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_ranking(response, CASE)


def test_failure_artifact_and_no_overwrite():
    client = FakeClient(answer(finish="length", content="session-key"))
    with TemporaryDirectory(dir=Path(__file__).parent) as root:
        out = Path(root) / "run"
        report = run_package(PACKAGE, out, api_key="session-key", client=client)
        saved = (out / "strata_results.json").read_text(encoding="utf-8")
        assert report["failed_count"] == 1 and report["success_count"] == 0
        assert report["results"][0]["raw_response"]["usage"]["total_tokens"] == 17
        assert report["results"][0]["timing"]["provider_call_wall_s"] >= 0
        assert "session-key" not in saved and "[REDACTED]" in saved
        with pytest.raises(ValueError, match="overwrite"):
            run_package(PACKAGE, out, api_key="session-key", client=client)
        with pytest.raises(ValueError, match="ranking length"):
            evaluate(PACKAGE, QRELS, {"strata": report})


def test_success_uses_existing_evaluator():
    with TemporaryDirectory(dir=Path(__file__).parent) as root:
        report = run_package(PACKAGE, Path(root) / "run", api_key="test-secret", client=FakeClient())
        assert report["success_count"] == 1 and report["failed_count"] == 0
        row = report["results"][0]
        assert row["cache_hit"] is False and row["timing"]["kind"] == "fresh"
        assert row["timing"]["e2e_s"] >= row["timing"]["provider_call_wall_s"]
        assert row["ranking"] == [{"candidate_id": "C01"}, {"candidate_id": "C02"}]
        metrics = evaluate(PACKAGE, QRELS, {"strata": report})
        assert metrics["providers"]["strata"]["per_query"]["Q01"]["ndcg_at_5"] == 1
        assert metrics["providers"]["strata"]["latency"]["fresh_e2e_sample_n"] == 1


def test_completed_cases_persist_before_next_request():
    second = json.loads(json.dumps(CASE))
    second["query_id"] = "Q02"
    package = {**PACKAGE, "cases": [CASE, second]}
    with TemporaryDirectory(dir=Path(__file__).parent) as root:
        out = Path(root) / "run"
        seen = []

        def progress(event, query_id, result):
            seen.append((event, query_id))
            if event == "start" and query_id == "Q02":
                first = json.loads((out / "cases" / "001.json").read_text(encoding="utf-8"))
                assert first["query_id"] == "Q01"
                assert (out / "runtime_metadata.json").exists()

        report = run_package(package, out, api_key="test-secret", client=FakeClient(), on_case=progress)
        assert report["success_count"] == 2
        assert seen == [("start", "Q01"), ("done", "Q01"),
                        ("start", "Q02"), ("done", "Q02")]


def test_http_exception_keeps_response_and_wall_time():
    from urllib.error import HTTPError
    from io import BytesIO

    class ErrorClient(FakeClient):
        def chat(self, messages, model, **kwargs):
            raise HTTPError("http://localhost", 500, "session-key failure", {},
                            BytesIO(b'{"message":"session-key refusal"}'))

    with TemporaryDirectory(dir=Path(__file__).parent) as root:
        out = Path(root) / "run"
        report = run_package(PACKAGE, out, api_key="session-key", client=ErrorClient())
        row = report["results"][0]
        assert row["status"] == "failed"
        assert row["http_error_response"]["status"] == 500
        assert row["timing"]["provider_call_wall_s"] >= 0
        assert "session-key" not in (out / "cases" / "001.json").read_text(encoding="utf-8")


def test_multiple_models_require_selection():
    class Multi(FakeClient):
        def get(self, path):
            if path == "/v1/models":
                return {"data": [{"id": "one"}, {"id": "two"}]}
            return super().get(path)

    with pytest.raises(ValueError, match="multiple"):
        discover(Multi())
    assert discover(Multi(), "two")["model"] == "two"
