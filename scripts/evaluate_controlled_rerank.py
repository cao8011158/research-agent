"""Evaluate frozen reranker rankings against independent graded qrels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.controlled_evaluation import evaluate


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--qrels", action="append", required=True, type=Path,
                        help="Repeat for disjoint curator files")
    parser.add_argument("--result", action="append", required=True,
                        help="Provider result as voyage=PATH, jev=PATH, or assistant=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    providers = {}
    for item in args.result:
        if "=" not in item:
            parser.error("--result must be NAME=PATH")
        name, path = item.split("=", 1)
        if name not in ("voyage", "jev", "assistant") or name in providers:
            parser.error("result NAME must be unique voyage, jev, or assistant")
        providers[name] = _read(path)
    report = evaluate(_read(args.cases), [_read(p) for p in args.qrels], providers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "providers": list(report["providers"]),
                      "query_count": len(next(iter(report["providers"].values()))["per_query"])}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"controlled evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
