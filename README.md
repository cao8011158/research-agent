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

Source code and notebooks live in this repository. Runtime code tests and GPU
acceptance run in Colab. Do not infer test success from the presence of a notebook
or installation command; consult the stage report for actual evidence.

Knowledge-graph retrieval, reranking, web search and planning experiments are
future stages and are not part of this deployment acceptance.
