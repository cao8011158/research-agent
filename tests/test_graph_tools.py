import asyncio
import time
import pytest

from research_agent.graph_tools import GraphTools, RELATIONS


class FakeTransaction:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        async def rows():
            for row in self.rows:
                yield row
        return rows()


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def execute_read(self, callback):
        assert 0 < callback.timeout <= 30
        return await callback(self.tx)


class FakeDriver:
    def __init__(self, rows):
        self.tx = FakeTransaction(rows)
        self.database = None

    def session(self, *, database):
        self.database = database
        return FakeSession(self.tx)


def run(coro):
    return asyncio.run(coro)


def call(tools, name, args):
    return run(tools.call(name, args, deadline=time.monotonic() + 30))


def test_get_node_uses_run_identity_and_omits_large_text():
    driver = FakeDriver([{"node_id": "p1", "labels": ["KGNode", "Paper"]}])
    result = call(GraphTools(driver, "graph:test", database="kg"), "get_node", {"node_id": "p1"})
    assert result["node_id"] == "p1"
    statement, params = driver.tx.calls[0]
    assert "run_id:$run_id,node_id:$node_id" in statement
    assert "n.text" not in statement
    assert params == {"run_id": "graph:test", "node_id": "p1"}
    assert driver.database == "kg"


@pytest.mark.parametrize("direction,pattern", [("out", "-[r:HAS_ITEM]->"), ("in", "<-[r:HAS_ITEM]-"), ("both", "-[r:HAS_ITEM]-")])
def test_expand_only_validated_one_hop(direction, pattern):
    driver = FakeDriver([])
    call(GraphTools(driver, "graph:test"), "expand_neighbors", {
        "node_id": "x", "relation_types": ["HAS_ITEM"], "direction": direction, "limit": 3})
    statement, params = driver.tx.calls[0]
    assert pattern in statement and "LIMIT $limit" in statement
    assert params["limit"] == 3


def test_invalid_relation_cannot_enter_query():
    driver = FakeDriver([])
    with pytest.raises(ValueError):
        call(GraphTools(driver, "graph:test"), "expand_neighbors", {
            "node_id": "x", "relation_types": ["HAS_ITEM) MATCH (x)"]})
    assert not driver.tx.calls


def test_shared_papers_requires_all_canonical_ids():
    driver = FakeDriver([])
    call(GraphTools(driver, "graph:test"), "find_shared_papers", {"node_ids": ["a", "b", "a"]})
    statement, params = driver.tx.calls[0]
    assert "SourceItem)-[:RESOLVES_TO]->(entity)" in statement
    assert "matched=$target_count" in statement
    assert params["node_ids"] == ["a", "b"] and params["target_count"] == 2
    assert "SOLVES" not in RELATIONS and "EVALUATED_ON" not in RELATIONS


def test_evidence_preserves_roles_source_and_lineage():
    row = {"requested_id": "m", "item_id": "m", "roles": ["used_method"],
           "source_json": "{\"section\":\"2\"}", "paper_id": "p", "version_id": "v"}
    driver = FakeDriver([row])
    result = call(GraphTools(driver, "graph:test"), "get_evidence", {"node_ids": ["m"], "limit": 1})
    assert result == [row]
    statement, params = driver.tx.calls[0]
    assert "HAS_ITEM" in statement and "HAS_SOURCE" in statement and "HAS_VERSION" in statement
    assert params["limit"] == 1
