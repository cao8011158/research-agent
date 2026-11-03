"""Run fresh Strata listwise ranking over a frozen case package."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from research_agent.strata_reranker import run_package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", help="Required if the server reports multiple loaded models")
    parser.add_argument("--reasoning-effort", default="none", choices=("none", "low", "medium", "high"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    key = os.environ.get("STRATA_API_KEY")
    if not key:
        parser.error("STRATA_API_KEY must be set in the environment")
    package = json.loads(args.input.read_text(encoding="utf-8"))
    def progress(event, query_id, result):
        if event == "start":
            print(f"Strata case {query_id}: starting", file=sys.stderr, flush=True)
        else:
            print(f"Strata case {query_id}: {result['status']} in {result['timing']['e2e_s']:.2f}s",
                  file=sys.stderr, flush=True)

    report = run_package(package, args.output_dir, base_url=args.base_url, api_key=key,
                         model=args.model, reasoning_effort=args.reasoning_effort,
                         max_tokens=args.max_tokens, timeout=args.timeout, on_case=progress)
    print(json.dumps({"results_file": str(args.output_dir / "strata_results.json"),
                      "success_count": report["success_count"],
                      "failed_count": report["failed_count"]}))
    return 1 if report["failed_count"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Strata reranker failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
