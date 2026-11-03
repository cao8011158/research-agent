import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from research_agent.mcp_tools import Neo4jMCPTools


ENV = {"NEO4J_MCP_URI": "bolt://localhost:7687", "NEO4J_MCP_USERNAME": "neo4j",
       "NEO4J_MCP_PASSWORD": "test-secret", "NEO4J_MCP_DATABASE": "neo4j"}


def run(coro):
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("unexpected suspended coroutine")


def test_requires_database_and_credentials_without_printing_values():
    with pytest.raises(ValueError) as err:
        Neo4jMCPTools(env={"NEO4J_MCP_URI": ENV["NEO4J_MCP_URI"]})
    assert "PASSWORD" in str(err.value)
    assert ENV["NEO4J_MCP_URI"] not in str(err.value)


def test_read_only_configuration_overrides_caller_flags():
    tools = Neo4jMCPTools(env={**ENV, "NEO4J_MCP_READ_ONLY": "false", "NEO4J_MCP_TOOLS": "write-cypher"})
    assert tools.command == "neo4j-mcp-server"
    assert tools.env["NEO4J_MCP_READ_ONLY"] == "true"
    assert tools.env["NEO4J_MCP_TOOLS"] == "get-schema,read-cypher"


def test_cli_cannot_override_read_only_server_configuration():
    with pytest.raises(ValueError):
        Neo4jMCPTools(args=["--read-only=false"], env=ENV)


def test_official_sdk_handshake_discovery_call_and_close(monkeypatch):
    calls = []

    class Context:
        async def __aenter__(self):
            calls.append("transport-enter")
            return "read", "write"

        async def __aexit__(self, *_):
            calls.append("transport-exit")

    class Session:
        def __init__(self, read, write):
            assert (read, write) == ("read", "write")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            calls.append("session-exit")

        async def initialize(self):
            calls.append("initialize")

        async def list_tools(self):
            calls.append("list")
            return SimpleNamespace(tools=[SimpleNamespace(name=name, description=name, inputSchema={"type": "object"})
                                          for name in ("get-schema", "read-cypher")])

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(isError=True, structuredContent={"reason": "rejected"},
                                   content=[SimpleNamespace(model_dump=lambda **_: {"type": "text", "text": "rejected"})])

    mcp = ModuleType("mcp")
    mcp.ClientSession = Session
    mcp.StdioServerParameters = lambda **kwargs: SimpleNamespace(**kwargs)
    client = ModuleType("mcp.client")
    stdio = ModuleType("mcp.client.stdio")
    stdio.stdio_client = lambda params: (calls.append(("params", params)) or Context())
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)

    async def exercise():
        async with Neo4jMCPTools(env=ENV) as adapter:
            assert {tool["function"]["name"] for tool in adapter.tools} == {
                "neo4j_get_schema", "neo4j_read_cypher"}
            return await adapter.call("neo4j_read_cypher", {"query": "RETURN 1"},
                                      deadline=time.monotonic() + 5)

    result = run(exercise())
    assert result["is_error"] is True
    assert result["content"] == [{"type": "text", "text": "rejected"}]
    assert result["structured_content"] == {"reason": "rejected"}
    assert ("read-cypher", {"query": "RETURN 1"}) in calls
    assert calls.index("initialize") < calls.index("list")
    assert calls[-2:] == ["session-exit", "transport-exit"]
    params = next(value for name, value in calls if name == "params")
    assert params.env["NEO4J_MCP_READ_ONLY"] == "true"


def test_rejects_write_tool_even_if_server_were_to_advertise_it():
    adapter = Neo4jMCPTools(env=ENV)
    with pytest.raises(ValueError):
        run(adapter.call("neo4j_write_cypher", {"query": "CREATE (n)"},
                         deadline=time.monotonic() + 5))
