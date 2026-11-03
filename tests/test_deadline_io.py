"""Real delayed loopback HTTP verifies no orphaned blocking worker hides elapsed time."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest
from jsonschema import validate  # Load validation before measuring cancellation.

from research_agent.harness import AgentHarness, Limits
from research_agent.llm import AsyncStrataClient


class SearchTool:
    calls = 0
    tools = [{"type": "function", "function": {"name": "search_nodes", "description": "Search",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"]}}}]

    async def call(self, name, args, *, deadline):
        self.calls += 1
        await asyncio.sleep(5)
        return {"results": []}


@contextmanager
def delayed_server(delay=2, *, tool_call=False):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            time.sleep(delay)
            message = ({"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "search_nodes", "arguments": '{"query":"x"}'}}]}
                       if tool_call else {"role": "assistant", "content": "late answer"})
            payload = json.dumps({"choices": [{"finish_reason": "tool_calls" if tool_call else "stop",
                                                 "message": message}]}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_delayed_actual_http_stops_at_deadline():
    async def exercise(url):
        async with AsyncStrataClient(url, "secret", timeout=1) as client:
            start = time.monotonic()
            trace = await AgentHarness(client, "model", [SearchTool()],
                                       limits=Limits(timeout_s=0.05)).run("question")
            return trace, time.monotonic() - start

    with delayed_server() as url:
        trace, wall = asyncio.run(exercise(url))
    assert trace["status"] == "incomplete" and trace["stop_reason"] == "timeout"
    assert trace["answer"] is None
    assert wall < 1.5 and trace["elapsed_s"] < 1.5
    assert abs(wall - trace["elapsed_s"]) < 0.5


def test_async_tool_cancellation_stops_at_deadline():
    class Client:
        calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            return {"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "search_nodes", "arguments": '{"query":"x"}'}}]}}]}

    start = time.monotonic()
    client = Client()
    tool = SearchTool()
    trace = asyncio.run(AgentHarness(client, "model", [tool],
                                     limits=Limits(timeout_s=0.05)).run("question"))
    wall = time.monotonic() - start
    assert trace["status"] == "incomplete" and trace["stop_reason"] == "timeout"
    assert wall < 4 and abs(wall - trace["elapsed_s"]) < 1.5
    assert trace["events"][-1]["result"]["error"]["type"] == "TimeoutError"
    assert client.calls == 1
    assert tool.calls == 1


def test_model_timeout_does_not_retry_after_deadline():
    class Client:
        calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            await asyncio.sleep(2)

    client = Client()
    trace = asyncio.run(AgentHarness(client, "model", [SearchTool()],
                                     limits=Limits(timeout_s=0.05, max_retries=3)).run("question"))
    assert trace["status"] == "incomplete" and trace["stop_reason"] == "timeout"
    assert client.calls == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_invalid_deadlines_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        Limits(timeout_s=value)
