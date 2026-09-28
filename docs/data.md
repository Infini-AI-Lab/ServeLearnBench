# Data

The `dataset/` directory of this repository contains the files that `slb export-data --out dataset` writes:
every scenario's task stream and environment, exactly as the code generates them. The same files are on
[Hugging Face](https://huggingface.co/datasets/haizhongzheng/ServeLearnBench). Running an evaluation does not
read them; the code builds the tasks in memory, and a test checks that the two agree.

```
data/<scenario>/serving.jsonl     the serving stream, in order
data/<scenario>/test.jsonl        the held-out test tasks of every window
environments/<scenario>/
    world.json                    the initial database (empty for Pitch)
    policy.md, workflow.md        the documents the agent can read (Retail, Banking)
    system_prompt.txt             the acting agent's system prompt
    tools.json                    tool schemas, `finish` included
    windows.json                  per window: first stream position and the Oracle bulletin
results/leaderboard.csv           per-setting scores of every model and method in the paper
fingerprints.json                 content hashes; `slb verify` checks the code against them
```

## Task fields

| Field | Meaning |
|---|---|
| `task_id` | unique within the scenario |
| `scenario`, `domain`, `tier` | e.g. `retail_l2`, `retail`, `L2` |
| `split` | `serving` or `test` |
| `window` | environment window index |
| `timestamp` | stream position shown to the agent |
| `category` | `hidden` or `fully_specified` |
| `instruction` | the user message exactly as the agent receives it |
| `evaluator` | scoring-only information, never shown to an agent: the ground-truth actions and answer (Retail, Banking), the customer and product attributes (Pitch), the policy family and slice |

The `evaluator` fields must not be given to an agent or a learning method.
