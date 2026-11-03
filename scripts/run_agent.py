"""Run one bounded local scholarly KG agent session into a fresh artifact directory."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from research_agent.harness import AgentHarness, Limits, BASE_PROMPT, COMPARISON_PROMPT, PROMPT_SHA256
from research_agent.llm import AsyncStrataClient
from research_agent.strata_reranker import _clean, discover_async


def _write_trace(path: Path, trace: dict) -> None:
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False),
                       encoding="utf-8")
    staging.replace(path)


def _source_commit() -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


async def run(args, *, on_event=None):
    key = os.environ["STRATA_API_KEY"]
    commit = _source_commit()
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(AsyncStrataClient(
            args.base_url, key, timeout=min(args.strata_timeout, args.timeout)))
        runtime = await discover_async(client, args.model)
        model = runtime["model"]
        providers = []
        if args.source_mode != "web":
            from kg_builder.bm25_retrieval import BM25Index
            from kg_builder.embeddings import VoyageEmbeddingModel
            from kg_builder.persistent_dense import PersistentDenseIndex
            from research_agent.retrieval_tools import (AsyncJevClient, AsyncVoyageQueryEncoder,
                                                        RetrievalTools)
            bm25 = BM25Index(args.bm25_index, expected_run_id=args.run_id)
            stack.callback(bm25.close)
            encoder = AsyncVoyageQueryEncoder(VoyageEmbeddingModel(max_concurrency=1))
            dense = PersistentDenseIndex(args.dense_index, expected_run_id=args.run_id,
                                         expected_graph_namespace=args.graph_namespace,
                                         expected_model_id=encoder.model_id)
            if args.reranker == "jev":
                rerank_client = await stack.enter_async_context(AsyncJevClient())
            else:
                rerank_client = client
            retrieval = RetrievalTools(bm25, dense, encoder, run_id=args.run_id,
                                       graph_namespace=args.graph_namespace,
                                       reranker=args.reranker, rerank_client=rerank_client,
                                       rerank_model=model if args.reranker == "strata" else None,
                                       filter_mode=args.jev_filter)
            providers.append(retrieval)
            if args.tool_mode in ("custom", "both"):
                from neo4j import AsyncGraphDatabase
                from research_agent.graph_tools import GraphTools
                driver = AsyncGraphDatabase.driver(os.environ["NEO4J_URI"],
                    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
                stack.push_async_callback(driver.close)
                providers.append(GraphTools(driver, args.graph_namespace, database=args.neo4j_database))
            if args.tool_mode in ("mcp", "both"):
                from research_agent.mcp_tools import Neo4jMCPTools
                providers.append(await stack.enter_async_context(Neo4jMCPTools()))
        if args.source_mode in ("web", "kg-web"):
            from research_agent.web_tools import WebTools
            providers.append(await stack.enter_async_context(WebTools(os.environ["TAVILY_API_KEY"])))
        harness = AgentHarness(client, model, providers, api_key=key, source_commit=commit,
                               system_prompt=COMPARISON_PROMPT if args.source_mode else BASE_PROMPT,
                               source_mode=args.source_mode,
                               extra_secrets=(os.environ["TAVILY_API_KEY"],)
                               if args.source_mode in ("web", "kg-web") else (),
                               limits=Limits(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
                                             max_retries=args.max_retries, timeout_s=args.timeout,
                                             max_result_chars=args.max_result_chars))
        trace = await harness.run(args.question, on_event=on_event)
    trace["runtime_metadata"] = _clean(runtime, key)
    if args.source_mode != "web":
        trace["index_metadata"] = {"bm25": bm25.manifest, "dense": dense.metadata}
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--run-id")
    parser.add_argument("--graph-namespace")
    parser.add_argument("--bm25-index", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("kg", "kg-web", "web"),
                        help="Evaluation evidence sources; omitted preserves the legacy KG prompt")
    parser.add_argument("--tool-mode", choices=("custom", "mcp", "both"), default="custom")
    parser.add_argument("--reranker", choices=("strata", "jev"), default="strata")
    parser.add_argument("--jev-filter", action="store_true",
                        help="Return only Jev relevant items with P(relevant) >= 0.70")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model")
    parser.add_argument("--strata-timeout", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-result-chars", type=int, default=32000)
    args = parser.parse_args(argv)
    if not os.environ.get("STRATA_API_KEY"):
        parser.error("STRATA_API_KEY is required in environment")
    if args.source_mode != "web" and any(getattr(args, name) is None for name in
                                         ("run_id", "graph_namespace", "bm25_index", "dense_index")):
        parser.error("KG modes require --run-id, --graph-namespace, --bm25-index and --dense-index")
    if args.source_mode in ("web", "kg-web") and not os.environ.get("TAVILY_API_KEY"):
        parser.error("TAVILY_API_KEY is required for web modes")
    if args.source_mode == "web" and any(getattr(args, name) is not None for name in
                                         ("run_id", "graph_namespace", "bm25_index", "dense_index")):
        parser.error("KG index and namespace options are not used in web mode")
    if args.jev_filter and args.reranker != "jev":
        parser.error("--jev-filter requires --reranker jev")
    if args.strata_timeout <= 0:
        parser.error("--strata-timeout must be positive")
    try:
        Limits(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
               max_retries=args.max_retries, timeout_s=args.timeout,
               max_result_chars=args.max_result_chars)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir.exists():
        parser.error("output directory must be fresh")
    args.output_dir.mkdir(parents=True)
    trace_path = args.output_dir / "trace.json"
    key = os.environ["STRATA_API_KEY"]
    prompt_hash = (hashlib.sha256(COMPARISON_PROMPT.encode("utf-8")).hexdigest()
                   if args.source_mode else PROMPT_SHA256)
    metadata = {"prompt_sha256": prompt_hash}
    if args.source_mode:
        metadata["source_mode"] = args.source_mode
    _write_trace(trace_path, _clean({"schema": "kg-agent-trace/1", "question": args.question,
                                    "status": "initializing", "answer": None, **metadata}, key))
    def progress(stage, trace, details):
        _write_trace(trace_path, trace)
        print(json.dumps({"event": stage, **details}, ensure_ascii=False),
              file=sys.stderr, flush=True)
    try:
        trace = asyncio.run(run(args, on_event=progress))
    except Exception as exc:
        trace = {"schema": "kg-agent-trace/1", "question": args.question,
                 "status": "initialization_failed", "answer": None,
                 "error": {"type": type(exc).__name__, "message": str(exc)}, **metadata}
        trace = _clean(_clean(trace, key), os.environ.get("TAVILY_API_KEY", ""))
    _write_trace(trace_path, trace)
    print(json.dumps({"status": trace["status"], "answer": trace["answer"],
                      "trace": str(trace_path)}, ensure_ascii=False))
    return 0 if trace["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
