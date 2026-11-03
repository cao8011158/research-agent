"""Offline artifact and repository-location checks for the executable entry point."""
import importlib.util
import json
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agent.py"
spec = importlib.util.spec_from_file_location("run_agent_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_commit_comes_from_script_repo_not_caller_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = subprocess.run(["git", "-C", str(SCRIPT.parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    assert cli._source_commit() == expected


def test_fresh_artifact_contains_sanitized_init_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("STRATA_API_KEY", "secret-123")
    async def fail(_args, **_kwargs):
        raise RuntimeError("discovery secret-123 failed")
    monkeypatch.setattr(cli, "run", fail)
    out = tmp_path / "fresh"
    code = cli.main(["Question?", "--run-id", "r", "--graph-namespace", "g",
                     "--bm25-index", str(tmp_path / "b"), "--dense-index", str(tmp_path / "d"),
                     "--output-dir", str(out)])
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    assert code == 2 and trace["status"] == "initialization_failed"
    assert "secret-123" not in json.dumps(trace)


def test_web_init_trace_and_prompt_match_other_explicit_arms(monkeypatch, tmp_path):
    monkeypatch.setenv("STRATA_API_KEY", "strata-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    async def fail(_args, **_kwargs):
        raise RuntimeError("tavily-secret discovery failed")
    monkeypatch.setattr(cli, "run", fail)
    hashes = set()
    for mode in ("kg", "kg-web", "web"):
        out = tmp_path / mode
        kg = (["--run-id", "r", "--graph-namespace", "g", "--bm25-index", "b",
               "--dense-index", "d"] if mode != "web" else [])
        assert cli.main(["Question?", "--source-mode", mode, *kg,
                         "--output-dir", str(out)]) == 2
        trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
        assert trace["source_mode"] == mode
        assert "tavily-secret" not in json.dumps(trace)
        hashes.add(trace["prompt_sha256"])
    assert len(hashes) == 1
