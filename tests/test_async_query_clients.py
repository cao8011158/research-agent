"""Async runtime transport contracts without external credentials or paid calls."""
import asyncio
import time
from types import SimpleNamespace

import httpx
import numpy as np

from research_agent.retrieval_tools import AsyncJevClient, AsyncVoyageQueryEncoder
from research_agent.reranker_experiment import jev_payload, run_jev
from research_agent.strata_reranker import discover_async


def test_voyage_official_async_query_role_and_validation(monkeypatch):
    from kg_builder.embeddings import VoyageEmbeddingModel
    import voyageai

    seen = {}
    class FakeAsyncClient:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        async def embed(self, texts, **kwargs):
            seen["texts"] = texts
            seen["embed"] = kwargs
            vector = [0.0] * 1024
            vector[0] = 1.0
            return SimpleNamespace(embeddings=[vector], total_tokens=3)

    monkeypatch.setenv("VOYAGE_API_KEY", "test-secret")
    monkeypatch.setattr(voyageai, "AsyncClient", FakeAsyncClient)
    encoder = AsyncVoyageQueryEncoder(VoyageEmbeddingModel(max_concurrency=1))
    result = asyncio.run(encoder.encode_query_async(["research question"],
                                                      deadline=time.monotonic() + 30))
    assert result.shape == (1, 1024) and result.dtype == np.float32
    assert seen["texts"] == ["research question"]
    assert seen["embed"] == {"model": "voyage-4-large", "input_type": "query",
                              "output_dimension": 1024, "output_dtype": "float", "truncation": False}
    assert seen["init"]["max_retries"] == 0


def test_async_discovery_preserves_loaded_model_selection():
    class Client:
        async def get(self, path):
            return {"/health": {"loaded": True},
                    "/v1/models": {"data": [{"id": "loaded-model", "status": "loaded"}]},
                    "/props": {"model_path": "local-model"},
                    "/metrics": {"engine": {}}}[path]

    result = asyncio.run(discover_async(Client()))
    assert result["model"] == "loaded-model"
    assert result["runtime"]["model_path"] == "local-model"


def test_jev_async_transport_reuses_shared_payload_and_validator(monkeypatch):
    case = {"query_id": "q", "query": "A?", "target_type": "paper",
            "criteria": "Evidence about A", "candidates": [{"candidate_id": "C01",
            "node_id": "p1", "type": "Paper", "text": "A method"}]}
    expected = jev_payload(case)
    seen = {}
    def handler(request):
        seen["payload"] = request.read().decode()
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"C01": {
            "type": "choice", "choice": "relevant", "confidence": 0.9,
            "probabilities": {"relevant": 0.9, "not_relevant": 0.1, "insufficient": 0.0}}}})

    monkeypatch.setenv("TYPESAFE_API_KEY", "local-fixture-key")
    async def exercise():
        client = AsyncJevClient()
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                          headers={"Authorization": "Bearer local-fixture-key"})
        async with client:
            raw = await client.decide(expected, deadline=time.monotonic() + 5)
        return run_jev(case, client, raw_response=raw)

    result = asyncio.run(exercise())
    assert seen["authorization"] == "Bearer local-fixture-key"
    import json
    assert json.loads(seen["payload"]) == expected
    assert result["accepted"] == ["C01"]
