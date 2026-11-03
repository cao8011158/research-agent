"""Frozen hybrid retrieval: two independent top-ten lists, then one reranking pool."""
from __future__ import annotations

import os
import time

from research_agent.reranker_experiment import run_jev, jev_payload
from research_agent.strata_reranker import parse_ranking, request_for

KINDS = ("Paper", "Method", "Benchmark", "ResearchProblem", "Contribution", "ResearchArea")


class AsyncVoyageQueryEncoder:
    """Official Voyage async query role with existing snapshot identity/shape checks."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.model_id = encoder.model_id
        self.query_model_id = encoder.query_model_id
        self.dimension = encoder.dimension

    async def encode_query_async(self, texts, *, deadline):
        import voyageai
        if len(texts) != 1 or not isinstance(texts[0], str) or not texts[0].strip():
            raise ValueError("one nonempty Voyage query is required")
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is required for Voyage encoding")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("agent time budget exhausted")
        client = voyageai.AsyncClient(api_key=key, max_retries=0,
                                      timeout=min(self.encoder.timeout, remaining))
        response = await client.embed(texts, model=self.encoder.model_name,
                                      input_type="query", output_dimension=self.dimension,
                                      output_dtype="float", truncation=False)
        return self.encoder._validate_result(response, 1)


class AsyncJevClient:
    """Async HTTP transport for the existing Jev payload and response validator."""

    def __init__(self, *, timeout=120):
        import httpx
        from kg_builder.decision_api import ENDPOINT
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY not set")
        self.model = "jev-1.14.0"
        self.endpoint = ENDPOINT
        self.client = httpx.AsyncClient(timeout=timeout,
                                        headers={"Authorization": f"Bearer {key}"})

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)

    async def decide(self, payload, *, deadline):
        if deadline - time.monotonic() <= 0:
            raise TimeoutError("agent time budget exhausted")
        response = await self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response.json()


class RetrievalTools:
    def __init__(self, bm25, dense, encoder, *, run_id: str, graph_namespace: str,
                 reranker="strata", rerank_client=None, rerank_model=None,
                 filter_mode=False, jev_threshold=0.70):
        self.bm25, self.dense, self.encoder = bm25, dense, encoder
        self.run_id, self.graph_namespace = run_id, graph_namespace
        self.reranker, self.rerank_client, self.rerank_model = reranker, rerank_client, rerank_model
        self.filter_mode, self.jev_threshold = filter_mode, jev_threshold
        self.last_audit = None
        bm = bm25.manifest
        dm = dense.metadata
        for label, manifest in (("BM25", bm), ("dense", dm)):
            if manifest.get("run_id") != run_id or manifest.get("graph_namespace") != graph_namespace:
                raise ValueError(f"{label} run or graph namespace differs")
        if bm.get("snapshot_hash") != dm.get("snapshot_hash"):
            raise ValueError("BM25 and dense snapshots differ")
        model_id = dm.get("model_id") or dm.get("document_model_id")
        if model_id != encoder.model_id or dm.get("dimension") != encoder.dimension:
            raise ValueError("dense document model or dimension differs")
        query_id = dm.get("query_model_id")
        if query_id is not None and query_id != encoder.query_model_id:
            raise ValueError("dense query model identity differs")
        if reranker not in ("strata", "jev"):
            raise ValueError("unknown reranker")
        if rerank_client is None:
            raise ValueError("rerank client required")
        if reranker == "strata" and not rerank_model:
            raise ValueError("Strata rerank model required")
        if filter_mode and reranker != "jev":
            raise ValueError("threshold filtering requires Jev")
        # One cold-start pass creates an ID-to-offset locator. Query hydration is O(hits).
        self._positions = {}
        for kind, corpus in bm25.records.items():
            for position in range(len(corpus)):
                node_id = corpus[position]["node_id"]
                if node_id in self._positions:
                    raise ValueError(f"duplicate node ID across frozen corpus: {node_id}")
                self._positions[node_id] = (kind, position)

    @property
    def tools(self):
        return [{"type": "function", "function": {"name": "search_nodes",
            "description": "Search the frozen scholarly graph nodes. Returns source text, stable IDs, and retrieval provenance. Use result IDs for graph tools; relevance ranking is not proof of an edge.",
            "parameters": {"type": "object", "additionalProperties": False,
                "properties": {"query": {"type": "string", "minLength": 1},
                    "node_types": {"type": "array", "minItems": 1,
                                   "items": {"type": "string", "enum": list(KINDS)}, "uniqueItems": True},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["query"]}}}]

    async def call(self, name: str, args: dict, *, deadline: float):
        if name != "search_nodes":
            raise ValueError("unknown retrieval tool")
        return await self.search_nodes(**args, deadline=deadline)

    async def search_nodes(self, query: str, node_types=None, top_k=5, *, deadline):
        self.last_audit = None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be 1..20")
        if node_types is not None and (not isinstance(node_types, list) or not node_types or
            any(not isinstance(k, str) or k not in KINDS for k in node_types) or
            len(node_types) != len(set(node_types))):
            raise ValueError("invalid node_types")
        selected = node_types or None
        if deadline - time.monotonic() <= 0:
            raise TimeoutError("agent time budget exhausted")
        lexical = self.bm25.search(query, types=selected, top_k=10)
        vector = (await self.encoder.encode_query_async([query], deadline=deadline))[0]
        if deadline - time.monotonic() <= 0:
            raise TimeoutError("agent time budget exhausted")
        semantic = self.dense.search(vector, types=selected, top_k=10)
        pool = {}
        for hit in semantic:
            node_id = hit["node_id"]
            if node_id not in self._positions:
                raise ValueError(f"dense node missing in BM25 corpus: {node_id}")
            kind, position = self._positions[node_id]
            if hit.get("kind", hit.get("type")) != kind:
                raise ValueError("dense node kind differs from BM25 corpus")
            source = self.bm25.records[kind][position]
            text = source["text"]
            pool[node_id] = {**source, "type": kind,
                             "dense_similarity": hit.get("similarity"), "bm25_score": None,
                             "origins": ["dense"]}
        for hit in lexical:
            node_id = hit["node_id"]
            if node_id in pool:
                pool[node_id]["origins"].append("bm25")
                pool[node_id]["bm25_score"] = hit["score"]
            else:
                if node_id not in self._positions:
                    raise ValueError(f"BM25 hit missing in frozen corpus: {node_id}")
                kind, position = self._positions[node_id]
                if hit["type"] != kind:
                    raise ValueError("BM25 hit kind differs from frozen corpus")
                source = self.bm25.records[kind][position]
                pool[node_id] = {**source, "type": kind, "bm25_score": hit["score"],
                                 "dense_similarity": None, "origins": ["bm25"]}
        if not pool:
            return {"query": query, "run_id": self.run_id, "graph_namespace": self.graph_namespace,
                    "reranker": self.reranker, "results": [], "candidate_count": 0}
        candidates = list(pool.values())
        case = {"query_id": "tool-search", "query": query, "target_type": "scholarly graph node",
                "criteria": "Evidence useful for answering the query, including explicit entities and constraints; a useful partial contribution may rank below a complete match.",
                "candidates": [{"candidate_id": f"C{i:02d}", "node_id": c["node_id"],
                                "type": c["type"], "text": c["text"]}
                               for i, c in enumerate(candidates, 1)]}
        if deadline - time.monotonic() <= 0:
            raise TimeoutError("agent time budget exhausted")
        if self.reranker == "strata":
            request = request_for(case, self.rerank_model, reasoning_effort="none")
            request_start = time.perf_counter()
            raw = await self.rerank_client.chat(request["messages"], self.rerank_model,
                                                temperature=0, reasoning_effort="none",
                                                max_tokens=request["max_tokens"])
            verdict = {"provider": "strata", "request_payload": request, "raw_response": raw,
                       "provider_call_wall_s": time.perf_counter() - request_start}
            self.last_audit = verdict
            verdict["ranking"] = parse_ranking(raw, case)
            verdict["status"] = "success"
            ranked = verdict["ranking"]
            score_kind = "listwise_rank_only"
        else:
            payload = jev_payload(case)
            request_start = time.perf_counter()
            raw = await self.rerank_client.decide(payload, deadline=deadline)
            request_wall = time.perf_counter() - request_start
            self.last_audit = {"provider": "jev", "request_payload": payload,
                               "raw_response": raw, "provider_call_wall_s": request_wall}
            verdict = run_jev(case, self.rerank_client, threshold=self.jev_threshold,
                              raw_response=raw)
            verdict["timing"]["provider_call_wall_s"] = request_wall
            verdict["timing"]["e2e_s"] += request_wall
            self.last_audit = verdict
            ranked = verdict["ranking"]
            if self.filter_mode:
                accepted = set(verdict["accepted"])
                ranked = [item for item in ranked if item["candidate_id"] in accepted]
            score_kind = "jev_p_relevant"
        by_id = {c["candidate_id"]: candidates[i] for i, c in enumerate(case["candidates"])}
        results = []
        for item in ranked[:top_k]:
            row = dict(by_id[item["candidate_id"]])
            raw_text = row["text"]
            row["text"] = raw_text[:4000]
            row["text_length"] = len(raw_text)
            row["text_truncated"] = len(raw_text) > 4000
            row["rerank_score"] = item.get("score")
            row["run_id"] = self.run_id
            row["graph_namespace"] = self.graph_namespace
            results.append(row)
        return {"query": query, "run_id": self.run_id, "graph_namespace": self.graph_namespace,
                "reranker": self.reranker, "score_semantics": score_kind,
                "candidate_count": len(pool), "results": results}
