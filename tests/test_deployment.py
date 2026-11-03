"""Run in Colab with the repository test suite; no GPU or server required."""

import importlib.util
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_pass_and_ram_gate():
    preflight = load_script("colab_preflight")
    gib = 1024**3
    disk = type("Disk", (), {"total": 150 * gib, "free": 120 * gib})()
    gpu = {"name": "A100", "vram_total_bytes": 80 * gib,
           "vram_free_bytes": 75 * gib, "driver": "example", "compute_capability": "8.0"}
    with patch.object(preflight, "meminfo", return_value={"MemTotal": 160 * gib, "MemAvailable": 140 * gib}), \
         patch.object(preflight, "gpu_info", return_value=[gpu]), \
         patch.object(preflight.shutil, "disk_usage", return_value=disk), \
         patch.object(preflight.platform, "system", return_value="Linux"), \
         patch.object(preflight.platform, "machine", return_value="x86_64"):
        report = preflight.inspect("/content")
        assert report["gate_passed"]
    with patch.object(preflight, "meminfo", return_value={"MemTotal": 160 * gib, "MemAvailable": 30 * gib}), \
         patch.object(preflight, "gpu_info", return_value=[gpu]), \
         patch.object(preflight.shutil, "disk_usage", return_value=disk), \
         patch.object(preflight.platform, "system", return_value="Linux"), \
         patch.object(preflight.platform, "machine", return_value="x86_64"):
        report = preflight.inspect("/content")
        assert not report["gate_passed"]
        assert not report["checks"]["ram_available_at_least_60_gb"]


def test_preflight_disk_gate():
    preflight = load_script("colab_preflight")
    gib = 1024**3
    disk = type("Disk", (), {"total": 150 * gib, "free": 80 * gib})()
    gpu = {"name": "A100", "vram_total_bytes": 80 * gib,
           "vram_free_bytes": 75 * gib, "driver": "example", "compute_capability": "8.0"}
    with patch.object(preflight, "meminfo", return_value={"MemTotal": 160 * gib, "MemAvailable": 140 * gib}), \
         patch.object(preflight, "gpu_info", return_value=[gpu]), \
         patch.object(preflight.shutil, "disk_usage", return_value=disk), \
         patch.object(preflight.platform, "system", return_value="Linux"), \
         patch.object(preflight.platform, "machine", return_value="x86_64"):
        report = preflight.inspect("/content")
        assert not report["gate_passed"]
        assert not report["checks"]["disk_free_at_least_100_gb"]


def test_smoke_choice_rejects_truncation():
    smoke = load_script("strata_smoke")
    try:
        smoke.choice({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    except AssertionError as exc:
        assert "finish_reason='length'" in str(exc)
    else:
        raise AssertionError("truncated reply was accepted")
    for reason in (None, "content_filter", "error"):
        try:
            smoke.choice({"choices": [{"finish_reason": reason, "message": {"content": "partial"}}]})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{reason!r} was accepted")


def test_smoke_tool_roundtrip_and_actual_drafter():
    smoke = load_script("strata_smoke")
    client = type("Client", (), {})()
    responses = {
        "/health": {"status": "ok", "loaded": True},
        "/v1/models": {"data": [{"id": "strata"}]},
        "/props": {"build_info": "Strata 0.1.56", "model_path": "/content/strata-data/Qwen3.8-Flash-Next/IQ3_S",
                   "is_sleeping": False, "default_generation_settings": {"n_ctx": 32768}},
        "/metrics": {"engine": {"drafting": "mtp", "model": "strata", "max_context": 32768}},
    }
    client.get = lambda path: responses[path]
    calls = []
    def chat(messages, model, **kwargs):
        calls.append((messages, model, kwargs))
        if len(calls) == 1:
            return {"choices": [{"finish_reason": "stop", "message": {"content": "A paper reports a result."}}]}
        if len(calls) == 2:
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "add_integers", "arguments": '{"a":3,"b":4}'}}]}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": "The sum is 7."}}]}
    client.chat = chat
    with patch.object(smoke, "StrataClient", return_value=client):
        result = smoke.run("http://127.0.0.1:8080", "key", 0)
    assert result["passed"]
    assert calls[2][0][-1] == {"role": "tool", "tool_call_id": "call_1", "name": "add_integers", "content": "7"}
    assert calls[1][2]["reasoning_effort"] == "medium"
    responses["/props"]["model_path"] = "/content/strata-data/Qwen3.8-Flash-Next/IQ2_XS"
    with patch.object(smoke, "StrataClient", return_value=client):
        try:
            smoke.run("http://127.0.0.1:8080", "key", 0)
        except AssertionError as exc:
            assert "Wrong model_path" in str(exc)
        else:
            raise AssertionError("wrong model was accepted")
    responses["/props"]["model_path"] = "/content/strata-data/Qwen3.8-Flash-Next/IQ3_S"
    responses["/metrics"]["engine"]["drafting"] = "dflash2"
    with patch.object(smoke, "StrataClient", return_value=client):
        try:
            smoke.run("http://127.0.0.1:8080", "key", 0)
        except AssertionError as exc:
            assert "drafter" in str(exc)
        else:
            raise AssertionError("wrong drafter was accepted")
    responses["/health"]["loaded"] = False
    with patch.object(smoke, "StrataClient", return_value=client):
        try:
            smoke.run("http://127.0.0.1:8080", "key", 0)
        except AssertionError as exc:
            assert "loaded model" in str(exc)
        else:
            raise AssertionError("unloaded model was accepted")
