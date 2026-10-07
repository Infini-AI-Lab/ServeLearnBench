# ServeLearnBench dataset

Every scenario's task stream and environment, as written by `slb export-data --out dataset`.
Layout and fields are described in [docs/data.md](../docs/data.md).

The `evaluator` fields of each task and `environments/*/windows.json` describe the hidden policy.
Never show them to an agent or a learning method.
