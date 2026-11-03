"""Check a running local Strata Chat Completions server (stdlib only)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError

from research_agent.llm import StrataClient


def choice(response: dict) -> dict:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("Chat response has no choices")
    selected = choices[0]
    reason = selected.get("finish_reason")
    if reason not in {"stop", "tool_calls"}:
        raise AssertionError(f"Chat did not complete normally: finish_reason={reason!r}")
    return selected


def run(base_url: str, key: str, wait_seconds: int, evidence: dict | None = None) -> dict:
    evidence = evidence if evidence is not None else {}
    client = StrataClient(base_url, key)
    evidence["requests"] = []
    def chat(messages: list[dict], model: str, **options: object) -> dict:
        evidence["requests"].append({"model": model, "messages": messages, **options})
        return client.chat(messages, model, **options)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            health = client.get("/health")
            models = client.get("/v1/models")
            break
        except (HTTPError, URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(10)
    items = models.get("data", [])
    evidence.update({"health": health, "models": models})
    if health.get("loaded") is not True or not items or not items[0].get("id") or items[0].get("status") == "unloaded":
        raise AssertionError("/v1/models did not report a loaded model")
    model = items[0]["id"]
    props = client.get("/props")
    metrics = client.get("/metrics")
    evidence.update({"props": props, "metrics": metrics})
    engine = metrics.get("engine", {})
    model_path = str(props.get("model_path") or "").lower().replace("-", "").replace("_", "")
    if "qwen3.8flashnext" not in model_path or "iq3s" not in model_path:
        raise AssertionError(f"Wrong model_path for Qwen3.8-Flash-Next IQ3_S: {props.get('model_path')!r}")
    if props.get("is_sleeping") is not False:
        raise AssertionError("Model is sleeping or unloaded")
    if props.get("build_info") != "Strata 0.1.56":
        raise AssertionError(f"Unexpected engine version: {props.get('build_info')!r}")
    context = props.get("default_generation_settings", {}).get("n_ctx")
    if not isinstance(context, int) or context < 32768 or engine.get("max_context", 0) < 32768:
        raise AssertionError(f"Context below 32K: props={context!r}, metrics={engine.get('max_context')!r}")
    if engine.get("model") != model:
        raise AssertionError(f"Engine model alias {engine.get('model')!r} differs from /v1/models {model!r}")
    if engine.get("drafting") != "mtp":
        raise AssertionError(f"Actual engine drafter is {engine.get('drafting')!r}, expected 'mtp'; note: {engine.get('drafting_note')!r}")
    simple = chat([{"role": "user", "content": "Reply with one sentence about a scientific paper."}],
                         model, max_tokens=512, reasoning_effort="none")
    evidence["chat"] = simple
    simple_choice = choice(simple)
    if not (simple_choice.get("message", {}).get("content") or "").strip():
        raise AssertionError("Chat returned empty content")

    tool = {"type": "function", "function": {
        "name": "add_integers", "description": "Add two integers and return their exact sum.",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}}
    first = chat([{"role": "user", "content": "Use add_integers to calculate 3 + 4, then report the result."}],
                        model, tools=[tool], max_tokens=4096, reasoning_effort="medium")
    evidence["tool_first"] = first
    first_choice = choice(first)
    assistant = first_choice.get("message", {})
    calls = assistant.get("tool_calls") or []
    if first_choice.get("finish_reason") != "tool_calls" or not calls:
        raise AssertionError("No completed tool call returned")
    if len(calls) != 1:
        raise AssertionError(f"Expected exactly one tool call; received {len(calls)}")
    call = next((c for c in calls if c.get("function", {}).get("name") == "add_integers"), None)
    if not call or not call.get("id"):
        raise AssertionError("Expected add_integers tool call with id")
    args = json.loads(call["function"]["arguments"])
    if args != {"a": 3, "b": 4} and args != {"a": 4, "b": 3}:
        raise AssertionError(f"Unexpected tool arguments: {args!r}")
    actual_sum = args["a"] + args["b"]
    continuation = chat([
            {"role": "user", "content": "Use add_integers to calculate 3 + 4, then report the result."},
            assistant,
            {"role": "tool", "tool_call_id": call["id"], "name": "add_integers", "content": str(actual_sum)},
        ], model, tools=[tool], max_tokens=4096, reasoning_effort="medium")
    evidence["tool_continuation"] = continuation
    final_choice = choice(continuation)
    answer = (final_choice.get("message", {}).get("content") or "").strip()
    if not answer or str(actual_sum) not in answer or final_choice.get("finish_reason") != "stop":
        raise AssertionError("Tool continuation did not produce a final answer containing 7")
    evidence.update({"passed": True, "model_id": model})
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", default="/content/strata-artifacts/smoke.json")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    key = os.environ.get("STRATA_API_KEY")
    if not key:
        parser.error("STRATA_API_KEY must be set in the Colab session")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {}
    try:
        result = run(args.base_url, key, args.wait_seconds, evidence)
    except Exception as exc:
        result = {**evidence, "passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Smoke {'PASS' if result['passed'] else 'FAIL'}; report: {output}")
    if not result["passed"]:
        print(f"{result['error_type']}: {result['error']}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
