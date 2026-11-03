import asyncio
import json

from research_agent.harness import AgentHarness, Limits, PROMPT_SHA256


class Provider:
    tools = [{"type": "function", "function": {"name": "search_nodes", "description": "Search",
              "parameters": {"type": "object", "additionalProperties": False,
                             "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return {"results": [{"node_id": "p1", "source_url": "https://source.example/p1"}]}


def response(content=None, calls=None, finish="stop"):
    return {"choices": [{"finish_reason": finish,
                         "message": {"role": "assistant", "content": content,
                                     **({"tool_calls": calls} if calls else {})}}]}


def call(name="search_nodes", arguments='{"query":"original entity"}', call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.inputs = []

    def chat(self, messages, model, **kw):
        self.inputs.append((json.loads(json.dumps(messages)), model, kw))
        return self.responses.pop(0)


def test_tool_continuation_and_grounded_answer():
    client = Client(response(calls=[call()], finish="tool_calls"),
                    response("Paper p1 (https://source.example/p1) supports this."))
    provider = Provider()
    trace = asyncio.run(AgentHarness(client, "loaded", [provider], api_key="secret").run("original question"))
    assert trace["status"] == "complete"
    assert trace["prompt_sha256"] == PROMPT_SHA256
    assert trace["search_queries"] == ["original entity"]
    assert provider.calls == [("search_nodes", {"query": "original entity"})]
    second = client.inputs[1][0]
    assert second[0]["role"] == "system" and second[1]["content"] == "original question"
    assert second[2]["tool_calls"][0]["id"] == "call_1"
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "call_1"
    assert client.inputs[0][2]["reasoning_effort"] == "medium"
    assert client.inputs[0][2]["tools"] == provider.tools


def test_bad_args_and_unknown_tool_are_returned_for_recovery():
    client = Client(response(calls=[call(arguments="{"), call(name="unknown", call_id="call_2")],
                             finish="tool_calls"), response("I cannot establish the relation."))
    provider = Provider()
    trace = asyncio.run(AgentHarness(client, "loaded", [provider]).run("question"))
    assert trace["status"] == "complete" and provider.calls == []
    assert all(not json.loads(m["content"])["ok"] for m in client.inputs[1][0][-2:])


def test_tool_budget_stops_without_answer():
    client = Client(response(calls=[call()], finish="tool_calls"))
    trace = asyncio.run(AgentHarness(client, "loaded", [Provider()],
                     limits=Limits(max_tool_calls=1, max_steps=1)).run("question"))
    assert trace["status"] == "incomplete" and trace["answer"] is None
    assert trace["stop_reason"] == "step_budget"


def test_secret_redacted_from_model_and_tool_trace():
    client = Client(response("The secret is token-xyz"))
    trace = asyncio.run(AgentHarness(client, "loaded", [Provider()], api_key="token-xyz").run("question"))
    assert "token-xyz" not in json.dumps(trace)


def test_malformed_batch_cannot_execute_partial_tool_calls():
    for bad_calls in ([call(), call(call_id="call_1")], [call(), call(call_id="")],
                      [call(), {"id": "call_2", "type": "function", "function": {"name": "search_nodes"}}]):
        provider = Provider()
        trace = asyncio.run(AgentHarness(Client(response(calls=bad_calls, finish="tool_calls")),
                                         "loaded", [provider]).run("question"))
        assert trace["status"] == "incomplete"
        assert trace["stop_reason"] == "malformed_tool_call_batch"
        assert provider.calls == []


def test_empty_final_answer_is_incomplete():
    trace = asyncio.run(AgentHarness(Client(response("  ")), "loaded", [Provider()]).run("question"))
    assert trace["status"] == "incomplete" and trace["answer"] is None


def test_progress_callbacks_receive_sanitized_checkpoints():
    client = Client(response("token-xyz"))
    checkpoints = []
    trace = asyncio.run(AgentHarness(client, "loaded", [Provider()], api_key="token-xyz").run(
        "question", on_event=lambda stage, snapshot, details: checkpoints.append((stage, snapshot, details))))
    assert [stage for stage, _, _ in checkpoints] == ["started", "model_start", "model_done", "finished"]
    assert all("token-xyz" not in json.dumps(snapshot) for _, snapshot, _ in checkpoints)
    assert checkpoints[-1][1] == trace
