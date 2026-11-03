# research-agent

Research experiments with a local Qwen3.8-Flash-Next model served by Strata.

The first stage implements model deployment and calls. The target is a Colab A100
80 GB GPU with high system RAM, the original ISTA-DASLab GSQ-RCO IQ3_S model,
Strata Terminal 0.2.10 / engine 0.1.56, and **MTP only**. DFlash2 is excluded.
The actual model and drafter must be checked after startup; requesting MTP alone
does not prove that it was loaded.

- [Colab deployment instructions](docs/COLAB.md)
- [Deployment and calls notebook](notebooks/01_strata_colab.ipynb)
- [Development, review and validation status](docs/reviews/model-deployment-20261103/STAGE_REPORT.md)
- [Scholarly KG harness and retrieval contract](docs/HARNESS.md)
- [Sequential Colab KG agent experiment](docs/COLAB_KG_AGENT.md)

Source code and notebooks live in this repository. Runtime code tests and GPU
acceptance run in Colab. Do not infer test success from the presence of a notebook
or installation command; consult the stage report for actual evidence.

The separate scholarly KG harness adds frozen dense/BM25 retrieval, listwise
reranking, and custom or official MCP graph reads. Its offline checks do not
establish Colab model, restored graph, MCP transport, or query quality acceptance.
Web search and optional planning or query reformulation skills are outside the
current harness.
