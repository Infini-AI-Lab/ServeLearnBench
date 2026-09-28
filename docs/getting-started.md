# Getting started

## Install

```bash
git clone https://github.com/haizhongzheng/ServeLearnBench.git
cd ServeLearnBench
pip install -e ".[dev]"     # Python 3.12+; dev adds pytest and ruff
```

Check that the nine scenarios regenerate exactly as evaluated in the paper (no API calls, a few minutes):

```bash
slb verify
pytest -q
```

Tasks, worlds and documents are generated deterministically by code. The same files are published as a dataset
for inspection or use in other frameworks ([data.md](data.md)).

## Models and endpoints

Any OpenAI-compatible chat-completions endpoint that supports native tool calling works.

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | model name as the endpoint expects it |
| `--base-url` | Fireworks | endpoint URL, e.g. `http://localhost:8000/v1` for a local vLLM or SGLang server |
| `--api-key-env` | `SLB_API_KEY` | environment variable that holds the key (local servers need none) |
| `--reasoning-effort` | `high` | sent as `reasoning_effort`; `none` omits it |

The agent sends no temperature or top-p, so each provider's defaults apply. `max_tokens` is 65,536 per call.

Pitch is graded by an LLM judge, configured separately:

| Option | Default |
|---|---|
| `--judge-model` | `accounts/fireworks/models/glm-5p3-flash` |
| `--judge-base-url` | Fireworks |
| `--judge-api-key-env` | `SLB_API_KEY` |
| `--judge-reasoning-effort` | `high` |

The paper's Pitch scores use this judge at temperature 0. A different judge gives scores that are not comparable
with the leaderboard.

## Running

```bash
slb run --scenario retail_l2 --method rag --model MODEL                 # one scenario
slb run --scenario retail_l1,retail_l2,retail_l3 --method blind --model MODEL
slb run --scenario pitch_l1 --method oracle --model MODEL --windows 0,1  # a subset of windows
slb run --scenario banking_l3 --method rag --model MODEL --resume        # continue an interrupted run
```

`--workers` sets how many episodes run in parallel (default 8). Output goes to
`results/<model>/<scenario>_<method>.jsonl` (`--out-dir`, or `--out` for one scenario).

`examples/run_all.sh` runs every scenario for the three included methods.

## Scoring

```bash
slb score results/MODEL/*.jsonl
slb score results/*/*.jsonl --json table.json
```

For each file the command reports the Hidden score and, for Retail and Banking, the Fully Specified score, then
Avg. H and Avg. F when all settings are present ([protocol.md](protocol.md#scoring)).

## Result files

The first line is `{"config": {...}}`. Every other line is one episode:

| Field | Meaning |
|---|---|
| `phase` | `serving` or `test` |
| `window`, `timestamp` | environment window and the stream position shown to the agent |
| `task_id`, `category` | the task and whether it is `hidden` or `fully_specified` |
| `reward` | 0 or 1 in Retail and Banking; the rubric score / 100 in Pitch |
| `answer_detail` | the verifier's breakdown (Pitch: per-attribute verdicts, penalties) |
| `tokens`, `judge_tokens` | the agent's and the judge's token usage for the episode |
| `trace` | every model turn: reasoning, reply, tool call, observation, per-turn usage |
| `record` | (learning methods, serving only) the episode text the method learns from |
