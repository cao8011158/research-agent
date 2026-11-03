# Colab scholarly KG agent experiment

Run these cells manually, in order, on the intended high RAM A100 Colab runtime. This is an experiment procedure, not a record of a completed Colab run. It combines the pinned PostgreSQL-free query bundle with the bounded research-agent harness. Keep the model download/cache in `/content/strata-data`; do not clear it or copy weights through Drive. The graph, indexes, and traces use local `/content` disk during the session. A Colab VM reset requires restoring the query bundle again.

The base prompt and fixed graph semantics are shared across `custom`, `mcp`, and `both`. Do not add planning, query reformulation, Cypher-writing skills, or web search to this comparison. Search uses `voyage-4-large` 1024-dimensional **query** encoding, BM25 Top10 plus dense Top10 union without RRF, and full-text Strata listwise reranking by default. Strata reranking uses `reasoning_effort=none`; the agent loop uses `medium`.

## Cell 1 — pin the exact published inputs

Fill the three values from the final KG/harness release report after publication. Use full 40-character KG and research-agent commit SHAs and the full 64-character bundle ZIP SHA-256. Do not substitute branch names, `main`, or the current notebook checkout. The guide stops if any value is still a placeholder.

```python
import os
KG_RUNTIME_REF = 'PASTE_40_CHARACTER_KG_COMMIT_FROM_RELEASE_REPORT'
RA_RUNTIME_REF = 'PASTE_40_CHARACTER_RA_COMMIT_FROM_RELEASE_REPORT'
KG_BUNDLE_SHA256 = 'PASTE_64_CHARACTER_BUNDLE_SHA256_FROM_RELEASE_REPORT'
for name, value, length in (('KG_RUNTIME_REF', KG_RUNTIME_REF, 40),
                            ('RA_RUNTIME_REF', RA_RUNTIME_REF, 40),
                            ('KG_BUNDLE_SHA256', KG_BUNDLE_SHA256, 64)):
    assert len(value) == length and all(c in '0123456789abcdef' for c in value), name
    os.environ[name] = value
```

## Cells 2–5 — restore the PG-free query runtime

Open `docs/COLAB_KG_RUNTIME.md` in the **pinned kg-builder commit** and execute its cells in this exact order. It is the source of the full install/restore commands, so this guide does not maintain a divergent copy:

1. Section 1: mount Drive; clone both repositories; detach at `KG_RUNTIME_REF` and `RA_RUNTIME_REF`; verify both `rev-parse HEAD` results; install the query-time Python packages. The install uses `kg-builder` with `--no-deps` and explicit BM25/HNSW/Voyage/Neo4j dependencies, so no PostgreSQL server is installed or started.
2. Section 2: copy `colab-query-bundle-v2.zip` from Drive to local disk, compare its streaming SHA-256 with `KG_BUNDLE_SHA256`, extract into a **fresh** `/content/kg-bundle`, and verify every nested manifest plus the frozen `run_id`, graph namespace, and snapshot hash. Do not run an index build or document embedding.
3. Section 3: start fresh Neo4j Community 5.26.32 on loopback port 17687 with its matching bundled APOC Core jar; set a runtime-only password; restore the published graph; verify graph counts/fingerprint and `apoc.meta.schema`. There is no PostgreSQL restore in this inference procedure.
4. Section 4: load complete BM25 corpus/index directories and the persistent six-kind HNSW/dense directories from the extracted bundle; check frozen identities and counts. Keep `manifest`, `KG_BUNDLE`, and the Neo4j environment variables from those cells in the notebook session.

The bundle guide's section 5 is the **official MCP handshake**, to run after the custom tool smoke below. Its tested Python entry point is `neo4j-mcp-server` from `neo4j-mcp-server==1.6.1`, with read-only `get-schema,read-cypher`. An `apoc.version()` response alone does not establish MCP readiness.

## Cell 6 — graph tools only, before any model search

This uses the restored graph but makes no Strata, Voyage, or Jev request. It checks a known paper ID within the frozen namespace. If the native read fails, stop before spending query API calls.

```python
import asyncio, time
from neo4j import AsyncGraphDatabase
from research_agent.graph_tools import GraphTools

async def custom_graph_smoke():
    driver = AsyncGraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']))
    try:
        tools = GraphTools(driver, manifest['graph_namespace'], database=os.environ['NEO4J_DATABASE'])
        paper = await tools.call('get_node', {'node_id': 'arxiv:2501.13956'},
                                 deadline=time.monotonic() + 30)
        assert paper is not None and paper['node_id'] == 'arxiv:2501.13956'
        print('Custom graph read:', paper['node_id'])
    finally:
        await driver.close()

await custom_graph_smoke()
```

Now run **section 5** of `docs/COLAB_KG_RUNTIME.md` to verify the actual official MCP initialize/list/call flow and the two read-only tools. Keep Neo4j running. Record errors from either mode separately; a mocked SDK test is not a transport pass.

## Cells 7–9 — recover or start the intended Strata model

Use the hardware preflight and Terminal installation cells from [`01_strata_colab.ipynb`](../notebooks/01_strata_colab.ipynb) if this VM has not done them. Reuse an already downloaded `/content/strata-data` and an already running verified server. Do not run the notebook's branch-based clone cell, nor its optional cleanup cell before the experiment. The launch request is original ISTA-DASLab Qwen3.8-Flash-Next IQ3_S, 32K context, one slot, text only, **MTP**, localhost **8081**, and no DFlash2. The old notebook's launch cell says 8080; use this corrected cell instead. Inspect the official plan before starting or reusing the service.

```python
from pathlib import Path
import secrets, subprocess
STRATA = str(Path.home() / '.local/bin/strata')
assert Path(STRATA).is_file(), 'Run the pinned Strata 0.2.10 installation cell first'
assert '0.2.10' in subprocess.check_output([STRATA, '--version'], text=True)
os.environ.setdefault('STRATA_API_KEY', secrets.token_urlsafe(32))
LAUNCH = [STRATA, 'launch', '--family', 'qwen', '--model', 'iq3_s',
          '--context', '32k', '--parallel', '1', '--vision', 'off',
          '--drafting', 'mtp', '--access', 'local', '--port', '8081',
          '--data-dir', '/content/strata-data', '--api-key-env', 'STRATA_API_KEY',
          '--output', 'json']
plan = subprocess.run(LAUNCH + ['--plan'], text=True, capture_output=True)
plan_text = plan.stdout + '\n' + plan.stderr
print(plan_text.replace(os.environ['STRATA_API_KEY'], '[REDACTED]'))
plan.check_returncode()
assert 'iq3_s' in plan_text.lower()
assert 'cuda13' not in plan_text.lower() and 'cuda 13' not in plan_text.lower()
```

```python
import urllib.request
BASE_URL = 'http://127.0.0.1:8081'
try:
    urllib.request.urlopen(BASE_URL + '/health', timeout=3).close()
    print('Existing Strata service responds on 8081; verifying it with smoke next')
except Exception:
    started = subprocess.run(LAUNCH + ['--yes', '--no-tui'], text=True,
                             capture_output=True)
    print((started.stdout + '\n' + started.stderr)
          .replace(os.environ['STRATA_API_KEY'], '[REDACTED]')[-12000:])
    started.check_returncode()
```

```python
smoke = subprocess.run(['python', '/content/research-agent/scripts/strata_smoke.py',
                        '--base-url', BASE_URL,
                        '--output', '/content/strata-artifacts/kg-agent-smoke.json'],
                       check=False)
smoke.check_returncode()
print('Actual model, 32K context, MTP and tool continuation verified by smoke')
```

The smoke makes model chat calls. It must confirm actual `/metrics.engine.drafting == 'mtp'`, model identity and context; the requested launch flags alone are insufficient. This smoke does **not** call Voyage or search the KG. No model search has run yet.

## Cell 10 — define a bounded query experiment

Set `VOYAGE_API_KEY` in the runtime's environment through Colab secrets or another private input. Every agent `search_nodes` call invokes the paid Voyage **query** API. `--reranker jev` would add paid Jev calls and requires `TYPESAFE_API_KEY`; keep that as a separate experiment. The cells above made neither call. The initial three sessions below use one natural-language task across `custom`, `mcp`, and `both`, with small step, tool, and time limits.

```python
MODES = ('custom', 'mcp', 'both')
QUESTIONS = (
    ('privacy-memory', 'Use search_nodes to find privacy-sensitive methods for long-term conversational memory. Trace any candidate through published paper and source evidence; separate supported facts from plausible claims.'),
    ('temporal-graph', 'Use search_nodes to find temporal or Graphiti-style memory architecture papers. Compare their stated contributions using source-linked evidence, and identify what the graph does not establish.'),
    ('scientific-api-benchmarks', 'Use search_nodes to find benchmarks for paper-grounded scientific API agents. Explain which benchmark descriptions and paper sources are actually present, including an empty result if none qualify.'),
    ('unsupported-relation', 'Search for a method and a research problem, then ask whether the graph proves a SOLVES or EVALUATED_ON edge between them. Cite only available evidence and state the relation boundary explicitly.'),
    ('fixed-paper', 'Inspect paper arxiv:2501.13956 and trace its published source evidence without inventing scientific relations.'),
    ('method-from-description', "Which published method uses configurable policies for people and events to control what an LLM agent can disclose from long-term memory? Identify the method and its paper, explain how the privacy controls work, and cite what the authors’ evaluation shows."),
    ('empty', 'Find a source-backed Method whose exact node ID is entity:does-not-exist; if absent, say so without inventing a match.'),
)
# Evaluator hints are kept outside the agent prompts; these are not gold answers.
REFERENCE_HINTS = {
    'fixed-paper': ('arxiv:2501.13956',),
    'method-from-description': ('arxiv:2609.35233',),
    'privacy-memory': ('area:0b98f6203c42760b4410820f92137641',),
    'scientific-api-benchmarks': ('entity:0014d3b42925cbe2170b2a71fed06811',),
}
assert os.environ.get('VOYAGE_API_KEY'), 'Set VOYAGE_API_KEY privately before continuing'
MAX_RUNS = 3  # One natural-language task across all three modes; extend after inspecting traces.
RUNS = [(mode, label, question) for label, question in QUESTIONS for mode in MODES][:MAX_RUNS]
print('First run:', len(RUNS), 'agent sessions; each has 4 steps, 6 tools, 180 seconds of harness time and a 240-second CLI limit')
```

The natural-language tasks exercise retrieval without giving the agent an exact ID. Reference hints can guide later human inspection but are **not gold answers**. The published graph has `HAS_VERSION`, `HAS_SOURCE`, `HAS_ITEM`, `RESOLVES_TO`, and `MEMBER_OF`; it has no `SOLVES` or `EVALUATED_ON`. A plausible research claim from retrieved prose is not an observed graph edge. Extend `MAX_RUNS` to include the explicit relation-boundary, public paper ID, method discovery, and absent-ID tasks after inspecting the first traces; retain empty/error outcomes rather than awarding correctness from a completed trace.

## Cell 11 — run the three tool modes with fresh trace directories

Run this cell only after Cell 10 passes. Each session gets a new output directory, a saved CLI log, and an exit code. Do not reuse or overwrite traces. The default Strata reranker is held constant for the mode comparison; Jev is a separate experiment with a separate output root. The CLI creates a sanitized initializing trace before runtime discovery and checkpoints model/tool events.

```python
import json, subprocess, time, uuid
from pathlib import Path
EXPERIMENT = Path('/content/kg-agent-experiment-' + uuid.uuid4().hex[:8])
EXPERIMENT.mkdir()
outcomes = []
for mode, label, question in RUNS:
    target = EXPERIMENT / f'{label}-{mode}'
    command = ['python', '/content/research-agent/scripts/run_agent.py', question,
               '--run-id', manifest['run_id'],
               '--graph-namespace', manifest['graph_namespace'],
               '--bm25-index', str(KG_BUNDLE / 'bm25'),
               '--dense-index', str(KG_BUNDLE / 'dense'),
               '--output-dir', str(target), '--tool-mode', mode,
               '--base-url', BASE_URL,
               '--max-steps', '4', '--max-tool-calls', '6',
               '--max-retries', '0', '--timeout', '180']
    start = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                env=os.environ.copy(), check=False, timeout=240)
        exit_code, output = result.returncode, result.stdout + '\n[stderr]\n' + result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = f'Whole CLI exceeded 240 seconds: {exc}'
    wall = time.monotonic() - start
    for secret_name in ('STRATA_API_KEY', 'VOYAGE_API_KEY', 'TYPESAFE_API_KEY', 'NEO4J_PASSWORD'):
        secret = os.environ.get(secret_name)
        if secret:
            output = output.replace(secret, '[REDACTED]')
    (EXPERIMENT / f'{label}-{mode}.log').write_text(output, encoding='utf-8')
    outcomes.append({'case': label, 'mode': mode, 'exit_code': exit_code,
                     'whole_cli_wall_s': wall, 'trace': str(target / 'trace.json')})
    print(label, mode, exit_code, round(wall, 2), 's')
(EXPERIMENT / 'outcomes.json').write_text(json.dumps(outcomes, indent=2))
```

## Cell 12 — summarize operational metrics, then grade evidence manually

```python
from collections import Counter
rows = []
for outcome in outcomes:
    trace_path = Path(outcome['trace'])
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {
        'status': 'no_trace', 'events': [], 'stop_reason': 'whole_cli_timeout'}
    events = trace.get('events', [])
    calls = [event for event in events if event.get('kind') == 'tool']
    errors = [event for event in events if event.get('kind') in ('model_error', 'tool_error')
              or (event.get('kind') == 'tool' and not event.get('result', {}).get('ok', True))]
    rows.append({**outcome, 'status': trace['status'],
                 'stop_reason': trace.get('stop_reason'),
                 'trace_elapsed_s': trace.get('elapsed_s'),
                 'tool_calls': len(calls), 'errors': len(errors),
                 'search_queries': len(trace.get('search_queries', []))})
print(json.dumps(rows, indent=2))
(EXPERIMENT / 'operational-summary.json').write_text(json.dumps(rows, indent=2))
```

Keep `whole_cli_wall_s` separate from `trace_elapsed_s` and from model or provider event times. Record startup, MCP handshake, restore, and index load as their own spans if measuring them. A timeout, provider error, empty result, or `initialization_failed` is an experiment outcome, not a missing row.

For each trace, a human reviewer records four separate judgments in a CSV or Markdown table: **scientific correctness**, **source/evidence grounding**, **problem decomposition**, and **recovery/uncertainty handling**. Use `pass / partial / fail / not assessable`, with cited node IDs, paper/source excerpts, and a reason. Review the raw answer against the frozen graph and source text; `status: complete` means the agent ended normally, not that the science is correct. For the absent-ID case, credit an explicit absence with no fabricated edge; inspect tool errors and how the agent responds. Preserve question, mode, commit pins, bundle SHA, model/engine/MTP metadata, traces, outcomes, and the human rubric together. Do not publish secrets, weights, or the full corpus as a trace artifact.

No local Colab GPU run, restored Neo4j/MCP handshake, paid Voyage query, or scientific-quality result is claimed by this guide. The operator must execute these cells and return the raw evidence before acceptance is reported.
