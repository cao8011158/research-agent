"""Bounded tool-calling loop for the frozen scholarly graph."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.resources import files

from research_agent.strata_reranker import _clean


BASE_PROMPT = files("research_agent").joinpath("prompts/base_system.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(BASE_PROMPT.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8
    max_tool_calls: int = 16
    max_retries: int = 1
    timeout_s: float = 600
    max_result_chars: int = 32000
    max_tokens: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.max_tool_calls,
                                                       self.max_result_chars, self.max_tokens)):
            raise ValueError("limits must be positive integers")
        if type(self.max_retries) is not int or self.max_retries < 0 or self.max_retries > 3:
            raise ValueError("max_retries must be 0..3")
        if (isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout_s must be finite and positive")


class AgentHarness:
    def __init__(self, client, model: str, providers, *, limits=Limits(), api_key="", source_commit=None):
        self.client, self.model, self.providers, self.limits = client, model, list(providers), limits
        self.api_key, self.source_commit = api_key, source_commit
        self.specs = []
        self.handlers = {}
        for provider in self.providers:
            for spec in provider.tools:
                name = spec["function"]["name"]
                if name in self.handlers:
                    raise ValueError(f"duplicate tool: {name}")
                self.handlers[name] = (provider, spec["function"]["parameters"])
                self.specs.append(spec)
        if not self.specs:
            raise ValueError("at least one tool is required")

    async def run(self, question: str, *, on_event=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty")
        started = time.monotonic()
        deadline = started + self.limits.timeout_s
        messages = [{"role": "system", "content": BASE_PROMPT},
                    {"role": "user", "content": question}]
        trace = {"schema": "kg-agent-trace/1", "question": question,
                 "model": self.model, "source_commit": self.source_commit,
                 "prompt_sha256": PROMPT_SHA256, "limits": asdict(self.limits),
                 "tools": self.specs, "events": [], "messages": messages,
                 "search_queries": [], "status": "incomplete", "answer": None}
        def emit(stage, **details):
            if on_event is not None:
                on_event(stage, _clean(trace, self.api_key), details)

        emit("started")
        tool_count = 0
        for step in range(self.limits.max_steps):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                trace["stop_reason"] = "timeout"
                break
            response = None
            model_timed_out = False
            for attempt in range(self.limits.max_retries + 1):
                if deadline - time.monotonic() <= 0:
                    break
                call_start = time.monotonic()
                emit("model_start", step=step, attempt=attempt)
                try:
                    response = await asyncio.wait_for(
                        self.client.chat(messages, self.model, tools=self.specs,
                                         max_tokens=self.limits.max_tokens,
                                         reasoning_effort="medium", temperature=0),
                        timeout=max(0.001, deadline - time.monotonic()))
                    trace["events"].append({"kind": "model", "step": step, "attempt": attempt,
                                            "elapsed_s": time.monotonic() - call_start,
                                            "raw_response": response})
                    emit("model_done", step=step, attempt=attempt)
                    break
                except asyncio.TimeoutError as exc:
                    model_timed_out = True
                    trace["events"].append({"kind": "model_error", "step": step, "attempt": attempt,
                                            "elapsed_s": time.monotonic() - call_start,
                                            "error": "TimeoutError: agent time budget exhausted"})
                    emit("model_error", step=step, attempt=attempt)
                    break
                except Exception as exc:
                    trace["events"].append({"kind": "model_error", "step": step, "attempt": attempt,
                                            "elapsed_s": time.monotonic() - call_start,
                                            "error": f"{type(exc).__name__}: {exc}"})
                    emit("model_error", step=step, attempt=attempt)
            if response is None:
                trace["stop_reason"] = "timeout" if model_timed_out or time.monotonic() >= deadline else "model_error"
                break
            choices = response.get("choices") if isinstance(response, dict) else None
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                trace["stop_reason"] = "malformed_model_response"
                break
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                trace["stop_reason"] = "malformed_model_response"
                break
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                trace["stop_reason"] = "malformed_model_response"
                break
            if tool_calls:
                ids = []
                for item in tool_calls:
                    if not isinstance(item, dict) or item.get("type") != "function":
                        break
                    fn = item.get("function")
                    if (not isinstance(fn, dict) or not isinstance(fn.get("name"), str)
                            or not isinstance(fn.get("arguments"), str)
                            or not isinstance(item.get("id"), str) or not item["id"]):
                        break
                    ids.append(item["id"])
                if (len(ids) != len(tool_calls) or len(ids) != len(set(ids))
                        or choice.get("finish_reason") != "tool_calls"):
                    trace["stop_reason"] = "malformed_tool_call_batch"
                    break
            messages.append({"role": "assistant", "content": message.get("content"),
                             **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                if (choice.get("finish_reason") == "stop" and isinstance(message.get("content"), str)
                        and message["content"].strip()):
                    trace["answer"] = message["content"]
                    trace["status"] = "complete"
                else:
                    trace["stop_reason"] = "unfinished_model_response"
                break
            if tool_count + len(tool_calls) > self.limits.max_tool_calls:
                trace["stop_reason"] = "tool_call_budget"
                break
            for tool_call in tool_calls:
                tool_count += 1
                call_start = time.monotonic()
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                fn = tool_call.get("function") if isinstance(tool_call, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                raw_args = fn.get("arguments") if isinstance(fn, dict) else None
                emit("tool_start", step=step, name=name)
                tool_timed_out = False
                try:
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("missing tool call ID")
                    if name not in self.handlers:
                        raise ValueError("unknown tool")
                    if not isinstance(raw_args, str):
                        raise ValueError("arguments must be JSON text")
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    from jsonschema import validate
                    provider, schema = self.handlers[name]
                    validate(args, schema)
                    if name == "search_nodes":
                        trace["search_queries"].append(args["query"])
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("agent time budget exhausted")
                    value = await asyncio.wait_for(provider.call(name, args, deadline=deadline),
                                                   timeout=remaining)
                    result = {"ok": not (isinstance(value, dict) and value.get("is_error") is True),
                              "data": value}
                except asyncio.TimeoutError:
                    tool_timed_out = True
                    result = {"ok": False, "error": {"type": "TimeoutError",
                              "message": "agent time budget exhausted"}}
                except Exception as exc:
                    result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
                encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
                if len(encoded) > self.limits.max_result_chars:
                    result = {"ok": False, "error": {"type": "ResultTooLarge",
                              "message": "Tool result exceeds max_result_chars; narrow the request."}}
                    encoded = json.dumps(result)
                trace["events"].append({"kind": "tool", "step": step, "name": name,
                                        "arguments": raw_args, "tool_call_id": call_id,
                                        "elapsed_s": time.monotonic() - call_start, "result": result,
                                        "rerank_audit": getattr(self.handlers[name][0], "last_audit", None)
                                        if name in self.handlers else None})
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name or "unknown",
                                 "content": encoded})
                emit("tool_done", step=step, name=name, ok=result["ok"])
                if tool_timed_out or time.monotonic() >= deadline:
                    trace["stop_reason"] = "timeout"
                    break
            if trace.get("stop_reason") == "timeout":
                break
        else:
            trace["stop_reason"] = "step_budget"
        trace["elapsed_s"] = time.monotonic() - started
        trace["tool_call_count"] = tool_count
        emit("finished", status=trace["status"])
        return _clean(trace, self.api_key)
