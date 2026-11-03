# Model deployment — 2026-11-03

## Current status

The user executes all Colab commands manually; remote control is cancelled. Deployment code candidate `aff5db417a1aac7ca3b205d4b5dc7c5f8091b5f8` is published on `codex/colab-strata-experiments`. The user reported **5 tests passed in 0.03 seconds, Preflight PASS, and real Smoke PASS** after changing the service and test URL to port **8081**. The basic deployment/call loop has therefore passed according to user-provided outputs. These are not independently reproduced runs; the remote `smoke.json` and logs have not yet been downloaded/reviewed. Retrieval quality, complex planning, performance and final independent acceptance remain untested. Earlier interrupted retrieval drafts remain untracked and excluded from delivery.

- Repository: https://github.com/cao8011158/research-agent
- Working branch: `codex/colab-strata-experiments`
- Baseline and current remote main last inspected: `75a08ee785a02a8b6d0c17a10501f6fe076fbd57`
- Implementation: deployment agent (`gpt-6-sol`, medium); integration/reporting and independent review: parent agent.
- Deployment code candidate: `aff5db417a1aac7ca3b205d4b5dc7c5f8091b5f8`, committed and pushed for validation; no merge authorization.
- Target runtime from user screenshots: A100 80 GB VRAM; approximately 167 GB system RAM; 368 GB local-scratch. These are screenshot observations, not a new runtime measurement.
- Target model: original ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S, MTP only. DFlash2 must not be accepted as the active drafter.
- Target release: Strata Terminal 0.2.10, Control and CUDA engine 0.1.56; actual installed versions and GPU architecture compatibility must be measured.
- Colab: https://colab.research.google.com/drive/1wVxRE3XB43SeWcuMGjxDkqLMPE6M7zmF

## Acceptance and test location

All runtime code tests and GPU tests run in Colab, per user instruction. Local work is limited to authoring, source review, and Git operations.

1. Python code tests for resource checks, HTTP client, launch configuration, and tool continuation.
2. Colab resource preflight: actual RAM, disk, GPU, driver, CPU and architecture; inspect the official Strata plan.
3. Install/start the intended model and verify actual model, quantization, engine, context and MTP identity.
4. Complete plain chat and a real tool-call/result/final-answer round trip; save inputs, outputs, timings, exit status and software versions without credentials.
5. Keep failure logs; do not mark a truncated or incomplete generation as passed.

Formal review rounds completed: 0/3. Source review before runtime validation is not a completed acceptance result.

## Stage 0 — scope and baseline

Read-only Git inspection confirmed the branch and baseline above. Prior incomplete drafts existed for deployment and wider experiments. Only deployment-related files will be included in the candidate commit. No tests, model downloads, Colab changes or paid model calls have been performed by this resumed stage.

Primary contracts: [Strata v0.2.10 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.10), [launch flags](https://github.com/Niko1221/Strata/blob/v0.2.10/docs/TERMINAL_LAUNCH.md), [CUDA package architectures](https://github.com/Niko1221/Strata/blob/v0.2.10/packaging/build_cuda_linux.sh), [server API](https://github.com/Niko1221/Strata/blob/v0.2.10/docs/DETAILS.md).

## Stage 1 — development candidate, uncommitted

The deployment agent delivered seven implementation/documentation/test files: `notebooks/01_strata_colab.ipynb`, `scripts/colab_preflight.py`, `scripts/strata_smoke.py`, `src/research_agent/llm.py`, `tests/test_deployment.py`, `tests/test_llm.py`, and `docs/COLAB.md`. Parent integration supplies the minimal project metadata, README, ignores and this report.

The candidate includes separate RAM/VRAM/disk checks, a reusable synchronous HTTP client, an ordinary chat with thinking disabled, and a one-tool-call/result/continuation check with medium reasoning. The smoke check consumes the shared client and requires actual `/metrics.engine.drafting == 'mtp'`; merely requesting MTP does not establish it. Completed API responses remain in failure evidence. The notebook separates code tests, preflight, installation, plan, start, smoke and optional stop.

Verification performed so far: PowerShell parsed notebook JSON (18 cells); source inspection checked upstream draft metrics against the implementation. No Python runtime, unit tests, model installs or API calls have run locally or remotely. Thus passed/failed/skipped runtime counts are **not available**, not zero failures. Candidate is not independently accepted or committed.

The user then selected Google's official Colab MCP as the preferred test connection. Connection setup is independent of model code validation. Official MCP needs local execution, Python >=3.13, and dynamic tool-list notifications. Existing Colab sessions and the supplied notebook must be preserved. No change to DFlash2 exclusion or model deployment scope.

## Stage 2 — DEPLOY-01 source fix, runtime validation pending

Independent pre-runtime source inspection found that an arbitrary loaded model could satisfy the initial smoke check. DEPLOY-01 (P2, this deployment stage) is corrected in the candidate: check the actual IQ3_S/Qwen3.8-Flash-Next model path, engine 0.1.56, loaded state, context >=32K, matching model alias, and MTP drafter. Only completed stop/tool_calls finish reasons are accepted. The tool result is computed from validated arguments, and request payloads, response evidence and timings are preserved. Regression cases were authored for wrong model, wrong drafter, unloaded model and bad finish reasons. No runtime tests have run; this is not a formal review pass.

The source-only ZIP excludes earlier retrieval drafts and credentials. It contains 11 deployment/project files plus a per-file SHA256 manifest; 14,444 bytes, SHA256 `082325ad6551e7ff60954284a50d4ea65c6b247bde433f5686dd0faf425dc428`. It is an uncommitted test candidate, not a released commit.

Google's official Colab MCP has been registered as `colab` in the local Codex MCP configuration with the official source pinned to `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`, using the existing uvx and Python 3.13 interpreter. `codex mcp get colab --json` confirmed the saved configuration. Bridge dependency installation is separate from project tests and does not load the model locally.

## Stage 3 — official MCP initialized; browser confirmation pending

- Official `colab-mcp --help` succeeded (exit 0) after dependency installation. Its package version from the inspected source is 1.0.1.
- A standard FastMCP client successfully initialized the official Google server and listed its `open_colab_browser_connection` tool. This is an MCP connection check, not a project test.
- Calling that tool opened a `scratchpad - Colab` Chrome page. The separate Computer Use request to read that Chrome window timed out at application approval. No UI controls were pressed.
- The official connection tool returned `structured_content.result: false`. A subsequent fresh `list_tools` still returned only the connection tool. Therefore the notebook is **not connected**; no notebook cells, files or runtime state have been read or modified through MCP.
- The user was asked to complete the MCP connection confirmation in the opened Colab page, or allow Computer Use access to Chrome if agent-assisted UI steps are wanted.
- The source-only deployment bundle remains ready; no project tests have run anywhere, no model has been downloaded, and no code has been committed or pushed. Formal review remains pending (0 completed rounds).

## Stage 4 — fresh official MCP connection attempt

The user confirmed the first connection page and reiterated the target notebook URL. A fresh tool listing still exposed only the connection tool. The expired local harness/server session was stopped normally and a fresh official MCP server was initialized successfully through the standard SDK. The connection tool opened a fresh page; the user confirmed that page during its approximately 60-second connection window. The tool nevertheless returned `structured_content.result: false` again at 2026-11-03 18:15 Asia/Shanghai. No notebook editing/execution tools have been received.

The local official MCP server log shows startup/list/call requests but does not establish a successful notebook connection. The exact cause remains unverified. Do not equate the user's successful GPU runtime connection or confirmation click with a completed MCP handshake. Do not modify Google's bridge or infer a confirmed upstream defect from source inspection alone.

The user was asked for the page's MCP/local-proxy status or error text, without a token-bearing URL. No Chrome Computer Use permission is required for the intended MCP workflow; no further Computer Use access was requested. Notebook modifications and all project tests remain pending this connection.

Only the `colab` MCP section's startup/tool timeouts were adjusted to 180/300 seconds after observing cold startup exceed the default 10 seconds. The original config was backed up as `config.toml.before-colab-timeouts-20261103`; unrelated MCP settings were preserved.

## Stage 5 — manual Colab execution; validation candidate

The user requested commands to execute themselves and revoked agent control of Colab. The active MCP control process exited 0 after `quit`. Do not reconnect or request Chrome control. The saved MCP registration remains configured; no installation was removed.

Two bounded static fixes were made before publication: the truncation regression now checks the actual `finish_reason='length'` error contract, and the preflight catches `subprocess.TimeoutExpired` from `nvidia-smi`. Parent source inspection confirmed these edits. Runtime tests remain **not run**, not passed.

Publication is explicitly a **validation candidate** for user-run Colab tests, not an accepted/released batch. The user's repository-based manual-test workflow requires making the source retrievable before test execution. No merge to main is authorized. The earlier ZIP is superseded by the published candidate because these two files changed.

Next user action: clone the feature branch at the supplied immutable commit, install the minimal test extra, run `pytest -q tests`, then `scripts/colab_preflight.py` in Colab. Inspect those actual outputs before downloading or starting the model. Formal review/acceptance stays pending.

## Stage 6 — candidate published; user commands supplied

Committed 12 files, 809 insertions and one deletion as `aff5db417a1aac7ca3b205d4b5dc7c5f8091b5f8` (`Add Strata MTP deployment candidate for Colab validation`). `git push -u origin codex/colab-strata-experiments` succeeded, exit 0. A subsequent `git ls-remote` confirmed the exact remote candidate SHA and unchanged main `75a08ee785a02a8b6d0c17a10501f6fe076fbd57`. No PR was opened and no merge occurred. This report update is a later documentation-only commit; the user-run commands deliberately pin the code candidate above.

Parent inspected the complete selected source files, notebook cells and authored tests. `git diff --cached --check` completed successfully before the candidate commit. Actual Colab unit test and preflight outputs are required next. No runtime pass/fail count can yet be reported. No independent runtime review has completed (0/3 completed rounds). Earlier retrieval drafts remain local and untracked, excluded from the remote commit.

## Stage 7 — user-reported Colab code tests and preflight

The user pasted outputs from the supplied checkout/test/preflight commands: `5 passed in 0.03s`; `Preflight PASS`. The output does not itself include the checkout SHA, so this is recorded as a report from the prescribed candidate commands, not an independently verified checkout or test run.

Reported runtime: Linux x86_64, glibc 2.39, 12 CPUs; system RAM total 179,370,455,040 bytes, available 176,534,327,296; disk free 252,949,053,440 bytes at `/content`; NVIDIA A100-SXM4-80GB, total VRAM 85,899,345,920 bytes, free 85,095,088,128; driver 580.82.07; compute capability 8.0; `CUDA_VERSION` 13.0.3. All capacity checks returned true. No GPU probe error. The remote preflight artifact is `/content/strata-artifacts/preflight.json`; that file has not been independently downloaded.

Next: install official Strata Terminal and inspect its version and IQ3_S/MTP launch plan. The runtime's CUDA_VERSION does not determine the engine package; the official component resolver checks driver and actual GPU architecture. The v0.2.10 CUDA 12 package covers sm_80. Do not downgrade the Colab driver or install a separate framework stack merely because CUDA_VERSION reports 13.

## Stage 8 — user-reported official launch plan

The user pasted a successful official plan (`schema_version: 1`, `ok: true`) for profile `launch-iq3_s-32k`. Settings: family qwen, model IQ3_S, context 32768, KV int8, vision off, parallel 1, drafting mtp, tuning skip, local access on port 8080, data directory `/content/strata-data`. No warnings. The planned download is 88.8 GB: main weights 83.6 GB plus MTP layer 5.2 GB; the missing engine is additional. The plan reports engine_missing true and engine_usable false, consistent with an engine not yet installed; it does not prove which engine will be acquired or that inference succeeds. The user did not paste Terminal version output, so the next startup instructions check it explicitly before the large download.

Next user-run operation: launch the planned model using the official command with `--yes --no-tui`, a session API key passed via its environment variable, and a redacted startup log. Then run the repository's real `strata_smoke.py` to verify the loaded model/engine/context/MTP and execute plain chat plus the tool round trip. No remote control or local model test is involved.

## Stage 9 — model prepared, port conflict, user-reported Smoke PASS

The user pasted Terminal output confirming `strata 0.2.10 (build 0.1.56; python 3.12.15)`. The official launcher downloaded, verified, test-started and installed `0.1.56 linux-x86_64-cuda-cuda12`, then prepared the IQ3_S model and MTP layer. The first service start failed because port 8080 was already occupied. No process was killed. The supplied correction changes the launch port to 8081 and the smoke URL to `http://127.0.0.1:8081`, reusing the prepared files.

The user subsequently reported `Smoke PASS; report: /content/strata-artifacts/smoke.json`. According to the inspected candidate script, a pass requires a loaded IQ3_S/Qwen3.8-Flash-Next model path, engine build `Strata 0.1.56`, at least 32K context, matching model aliases, actual drafter `mtp`, a nonempty ordinary chat, an actual `add_integers` tool call with validated arguments, and a completed final answer after the computed tool result is returned. This establishes the basic loop at the script's bounded acceptance level as reported by the user; it does not establish complex planning, retrieval quality, broader stability, or a measured speed target.

The remote JSON and startup logs remain in Colab and have not been independently retrieved. Next user commands should archive the small evidence directory and provide a normal client usage example on port 8081. Keep source code candidate and runtime port recorded; do not silently treat the notebook's original 8080 default as the successful configuration. No main merge is authorized.
