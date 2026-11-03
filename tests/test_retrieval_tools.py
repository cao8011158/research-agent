from research_agent.retrieval_tools import RetrievalTools


class Corpus:
    def __init__(self, rows):
        self.rows = rows
        self.reads = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads += 1
        return self.rows[index]


class BM:
    manifest = {"run_id": "r", "graph_namespace": "g", "snapshot_hash": "s"}

    def __init__(self):
        self.records = {"Paper": Corpus([{"node_id": "p1", "text": "paper one" + "x" * 5000, "source_url": "u1"},
                                          {"node_id": "p2", "text": "paper two", "source_url": "u2"}])}
        self.queries = []

    def search(self, query, *, types, top_k):
        self.queries.append((query, types, top_k))
        return [{"node_id": "p2", "type": "Paper", "text": "paper two", "source_url": "u2", "score": 4.0}]


class Dense:
    metadata = {"run_id": "r", "graph_namespace": "g", "snapshot_hash": "s",
                "model_id": "doc", "query_model_id": "query", "dimension": 2}

    def search(self, vector, *, types, top_k):
        assert list(vector) == [1, 0] and top_k == 10
        return [{"node_id": "p1", "kind": "Paper", "similarity": 0.8}]


class Encoder:
    model_id, query_model_id, dimension = "doc", "query", 2

    def encode_query(self, queries):
        assert queries == ["method question"]
        return [[1, 0]]

    def encode(self, _):
        raise AssertionError("document encoding must never run")


class RankClient:
    def chat(self, messages, model, **kwargs):
        import json
        candidates = json.loads(messages[1]["content"])["candidates"]
        assert len(candidates) == 2
        assert len(candidates[0]["text"]) > 4000
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"ranking": [c["candidate_id"] for c in reversed(candidates)]})}}],
            "model": model}


def test_union_preserves_bm25_only_hit_and_hydrates_by_id_once():
    bm = BM()
    retrieval = RetrievalTools(bm, Dense(), Encoder(), run_id="r", graph_namespace="g",
                               rerank_client=RankClient(), rerank_model="loaded")
    cold_reads = bm.records["Paper"].reads
    out = retrieval.search_nodes("method question", node_types=["Paper"])
    assert bm.queries == [("method question", ["Paper"], 10)]
    assert out["candidate_count"] == 2
    assert [r["node_id"] for r in out["results"]] == ["p2", "p1"]
    assert out["results"][0]["origins"] == ["bm25"]
    assert out["results"][1]["origins"] == ["dense"]
    assert out["results"][1]["source_url"] == "u1"
    assert bm.records["Paper"].reads == cold_reads + 2
    assert out["results"][1]["text_truncated"] is True
    assert len(out["results"][1]["text"]) == 4000
    assert out["score_semantics"] == "listwise_rank_only"
