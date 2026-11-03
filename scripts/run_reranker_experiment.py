"""Run or exchange frozen reranker pilot cases. Set PYTHONPATH to both project src roots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.reranker_experiment import (
    import_assistant, neutral_packet, run_package, validate_package,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("voyage", "jev", "export-assistant", "import-assistant"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assistant-results", type=Path, help="JSON list of assistant result objects for import")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--force", action="store_true", help="make fresh calls even when exact cache exists")
    args = parser.parse_args(argv)
    package = validate_package(_read(args.input))
    if args.command in ("voyage", "jev"):
        summary = run_package(package, args.command, args.output_dir,
                              threshold=args.threshold, force=args.force)
        print(json.dumps({"provider": args.command, "run_id": package["run_id"],
                          "fresh_count": summary["fresh_count"],
                          "cache_hit_count": summary["cache_hit_count"],
                          "results_file": str(args.output_dir / f"{args.command}_results.json")}))
    elif args.command == "export-assistant":
        paths = []
        for case in package["cases"]:
            path = args.output_dir / "assistant_packets" / f"{case['query_id']}.json"
            _write(path, neutral_packet(case))
            paths.append(str(path))
        print(json.dumps({"packets": paths}))
    else:
        if args.assistant_results is None:
            parser.error("--assistant-results required for import-assistant")
        responses = _read(args.assistant_results)
        if not isinstance(responses, list) or len(responses) != len(package["cases"]):
            raise ValueError("assistant results must be one result per case")
        by_id = {r.get("query_id"): r for r in responses if isinstance(r, dict)}
        if len(by_id) != len(responses) or set(by_id) != {c["query_id"] for c in package["cases"]}:
            raise ValueError("assistant results must have unique, matching query IDs")
        results = [import_assistant(c, by_id[c["query_id"]]) for c in package["cases"]]
        path = args.output_dir / "assistant_results.json"
        _write(path, {"schema": package["schema"], "run_id": package["run_id"], "results": results})
        print(json.dumps({"results_file": str(path), "count": len(results)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"reranker experiment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
