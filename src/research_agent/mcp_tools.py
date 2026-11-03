"""Persistent official Neo4j MCP stdio client for the agent tool registry."""
from __future__ import annotations

from contextlib import AsyncExitStack
import os
from collections.abc import Mapping

PUBLIC_NAMES = {"neo4j_get_schema": "get-schema", "neo4j_read_cypher": "read-cypher"}
REQUIRED_ENV = ("NEO4J_MCP_URI", "NEO4J_MCP_USERNAME", "NEO4J_MCP_PASSWORD", "NEO4J_MCP_DATABASE")


class Neo4jMCPTools:
    """One handshake and tool discovery per harness run; no homegrown protocol."""

    def __init__(self, *, command: str = "neo4j-mcp-server", args: list[str] | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        self.command = command
        self.args = list(args or [])
        controlled_flags = ("--read-only", "--tools", "--transport-mode", "--uri", "--username", "--password", "--database")
        if any(arg == flag or arg.startswith(flag + "=") for arg in self.args for flag in controlled_flags):
            raise ValueError("Neo4j MCP connection and read-only settings must use the configured environment")
        self.env = dict(env) if env is not None else self._environment()
        missing = [key for key in REQUIRED_ENV if not self.env.get(key)]
        if missing:
            raise ValueError("Neo4j MCP requires: " + ", ".join(missing))
        # The server enforces read-only classification and advertises only these two tools.
        self.env.update({"NEO4J_MCP_READ_ONLY": "true", "NEO4J_MCP_TOOLS": "get-schema,read-cypher",
                         "NEO4J_MCP_TELEMETRY": "false", "NEO4J_MCP_TRANSPORT_MODE": "stdio"})
        self._stack: AsyncExitStack | None = None
        self._session = None
        self.tools: list[dict] = []

    @staticmethod
    def _environment() -> dict[str, str]:
        env = {}
        for suffix in ("URI", "USERNAME", "PASSWORD", "DATABASE"):
            key = "NEO4J_MCP_" + suffix
            env[key] = os.environ.get(key) or os.environ.get("NEO4J_" + suffix, "")
        return env

    async def __aenter__(self) -> "Neo4jMCPTools":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(
                StdioServerParameters(command=self.command, args=self.args, env=self.env)))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            if names != set(PUBLIC_NAMES.values()):
                raise RuntimeError("Neo4j MCP advertised tools outside the expected read-only pair")
            self.tools = [{"type": "function", "function": {
                "name": public, "description": next(t.description or "" for t in listed.tools if t.name == server),
                "parameters": next(t.inputSchema for t in listed.tools if t.name == server)}}
                for public, server in PUBLIC_NAMES.items()]
            self._session = session
            self._stack = stack
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        self.tools = []
        if stack is not None:
            await stack.__aexit__(exc_type, exc_value, traceback)

    async def call(self, name: str, args: Mapping) -> dict:
        if name not in PUBLIC_NAMES:
            raise ValueError(f"unknown Neo4j MCP tool: {name}")
        if self._session is None:
            raise RuntimeError("Neo4j MCP session is not started")
        if not isinstance(args, Mapping):
            raise ValueError("tool arguments must be an object")
        result = await self._session.call_tool(PUBLIC_NAMES[name], arguments=dict(args))
        # Preserve the SDK's error flag, typed content, and structured result. A
        # server error must reach the model as an error, never as a fabricated answer.
        return {"is_error": result.isError,
                "content": [block.model_dump(mode="json", exclude_none=True) for block in result.content],
                "structured_content": result.structuredContent}
