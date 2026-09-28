<div align="center">

# ServeLearnBench

**How well can agents self-improve from serving experience?**

Haizhong Zheng<sup>1</sup>, Yizhuo Di<sup>1</sup>, Ranajoy Sadhukhan<sup>1</sup>, Shuowei Jin<sup>2</sup>, Beidi Chen<sup>1</sup>

<sup>1</sup>Carnegie Mellon University &nbsp; <sup>2</sup>Amazon

[![Website](https://img.shields.io/badge/website-online-9cf.svg)](https://haizhongzheng.github.io/ServeLearnBench/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow.svg)](https://huggingface.co/datasets/haizhongzheng/ServeLearnBench)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

</div>

The knowledge an agent needs in deployment is often implicit, undisclosed, and subject to change. ServeLearnBench
measures whether an agent can **infer that knowledge from its own serving experience, revise it when it goes stale,
and keep behaving correctly on everything that did not change**.

Each scenario is an *evolving-environment streaming dataset* (EESD): a chronological stream of tasks split into
environment windows. A hidden policy governs each window and changes between windows. The agent never sees the
policy or any correction; after each serving task it receives only a score. Held-out test tasks at the end of every
window measure what it has learned.

<p align="center"><img src="docs/assets/overview.png" width="92%" alt="The ServeLearnBench protocol"></p>

| | Retail | Banking | Pitch |
|---|---|---|---|
| Task | multi-turn customer support over an order database | case review: card transactions, limit increases, transfers | a 40–80 word sales pitch from a product sheet |
| Hidden knowledge | categorical store rules | numeric thresholds, caps and corridors | the current customer's taste |
| Grading | final database state or the right refusal code | the decision and its reason code | LLM-judged rubric (coverage, penalties for disliked, false or invented claims) |
| Tiers | L1 acquisition · L2 revision and composition · L3 non-monotonic change | same | same |

Nine scenarios, **53 windows, 4,508 serving tasks and 3,210 held-out test tasks**.

## Installation

```bash
git clone https://github.com/haizhongzheng/ServeLearnBench.git
cd ServeLearnBench
pip install -e .            # Python 3.12+
slb verify                  # regenerates all nine scenarios and checks them against the paper's
```

## Quick start

Any OpenAI-compatible endpoint with native tool calling works. The key is read from `SLB_API_KEY`.

```bash
export SLB_API_KEY=...

# Blind and Oracle references on one scenario (test tasks only)
slb run --scenario banking_l1 --method blind  --model accounts/fireworks/models/glm-5p3-flash
slb run --scenario banking_l1 --method oracle --model accounts/fireworks/models/glm-5p3-flash

# a learning method: the full serving stream, then the test sets of every window
slb run --scenario banking_l1 --method rag --model accounts/fireworks/models/glm-5p3-flash

# setting scores and averages, as reported in the paper
slb score results/*/*.jsonl
```

Results are written to `results/<model>/<scenario>_<method>.jsonl`, one line per episode with the full trajectory.
Pitch is scored by an LLM judge (`--judge-model`, default `glm-5p3-flash` at high reasoning effort, as in the paper).
See [docs/getting-started.md](docs/getting-started.md) for all options and local servers.

## Methods

| Method | Learns | Description |
|---|:---:|---|
| `blind` | – | the domain prompt and the task only; no serving experience |
| `oracle` | – | additionally told the hidden policy of the current window |
| `rag` | ✓ | BM25 retrieval of the agent's own past episodes (with their scores) into the prompt |

A new method implements two hooks, `context()` and `update()`; see [docs/methods.md](docs/methods.md).
The paper also evaluates Mem0, SkillOpt, Continual Harness and Prime; their results are in the leaderboard below.

## Leaderboard

Avg. H is the mean Hidden-Dependent score over the nine settings, Avg. F the mean Fully Specified score over the six
Retail and Banking settings; every setting counts once. API cost is the total over the nine settings (serving stream,
learning work and held-out test). Per-setting scores are in [docs/leaderboard.csv](docs/leaderboard.csv).

| # | Model | Method | Avg. H | Avg. F | API cost ($) |
|---:|---|---|---:|---:|---:|
| 1 | Opus 5 | CH | 73.4 | 96.2 | 2,449 |
| 2 | GLM-5.3 | CH | 69.8 | 96.7 | 1,077 |
| 3 | GLM-5.3 | Prime | 63.7 | 95.6 | 2,930 |
| 4 | Kimi K3 | Prime | 61.7 | 93.8 | 2,479 |
| 5 | GLM-5.3 Flash | Prime | 61.1 | 95.2 | 172 |
| 6 | DeepSeek V4.1 Flash | Prime | 60.3 | 92.4 | 429 |
| 7 | Kimi K3 | CH | 57.8 | 84.2 | 1,335 |
| 8 | GPT-5.6 Terra | CH | 55.8 | 94.2 | 239 |
| 9 | DeepSeek V4.1 Flash | CH | 53.3 | 91.3 | 52 |
| 10 | Kimi K3 | SkillOpt | 51.1 | 84.3 | 605 |
| 11 | Opus 5 | SkillOpt | 50.4 | 92.7 | 629 |
| 12 | GLM-5.3 Flash | CH | 49.7 | 82.8 | 89 |
| 13 | Opus 5 | Mem0 | 47.8 | 90.3 | 2,003 |
| 14 | GLM-5.3 | RAG | 45.9 | 87.6 | 932 |
| 15 | Kimi K3 | RAG | 45.0 | 89.4 | 1,073 |
| 16 | DeepSeek V4.1 Flash | RAG | 44.3 | 83.3 | 192 |
| 17 | GLM-5.3 | Mem0 | 43.7 | 95.9 | 565 |
| 18 | Opus 5 | RAG | 40.9 | 88.1 | 2,602 |
| 19 | GLM-5.3 | SkillOpt | 39.5 | 91.4 | 251 |
| 20 | GLM-5.3 Flash | Mem0 | 38.4 | 92.7 | 49 |
| 21 | GLM-5.3 Flash | RAG | 38.2 | 91.1 | 56 |
| 22 | Kimi K3 | Mem0 | 37.7 | 93.3 | 829 |
| 23 | GLM-5.3 Flash | SkillOpt | 35.4 | 85.1 | 35 |
| 24 | GPT-5.6 Terra | RAG | 34.9 | 91.4 | 496 |
| 25 | DeepSeek V4.1 Flash | Mem0 | 33.5 | 94.3 | 76 |
| 26 | DeepSeek V4.1 Flash | SkillOpt | 33.2 | 87.1 | 40 |
| 27 | GPT-5.6 Terra | Mem0 | 31.7 | 96.1 | 206 |
| 28 | GPT-5.6 Terra | SkillOpt | 29.6 | 94.6 | 146 |
| | *references* | | | | |
| | GPT-5.6 Terra | Blind | 15.1 | 97.8 | 40 |
| | GLM-5.3 Flash | Blind | 14.6 | 96.7 | 7 |
| | Opus 5 | Blind | 14.5 | 79.7 | 207 |
| | DeepSeek V4.1 Flash | Blind | 14.4 | 91.1 | 11 |
| | GLM-5.3 | Blind | 13.9 | 95.3 | 40 |
| | Kimi K3 | Blind | 12.3 | 98.0 | 120 |
| | GPT-5.6 Terra | Oracle | 97.6 | 98.6 | 36 |
| | Opus 5 | Oracle | 97.1 | 100.0 | 131 |
| | DeepSeek V4.1 Flash | Oracle | 95.6 | 99.9 | 7 |
| | Kimi K3 | Oracle | 95.5 | 99.8 | 102 |
| | GLM-5.3 | Oracle | 93.6 | 99.0 | 29 |
| | GLM-5.3 Flash | Oracle | 93.1 | 98.5 | 5 |

<p align="center"><img src="docs/assets/frontier.png" width="92%" alt="Hidden reward against token and API cost per task"></p>

## Documentation

- [Getting started](docs/getting-started.md): installation, endpoints, running and scoring
- [Benchmark](docs/benchmark.md): the EESD formulation, domains, tiers and scenarios
- [Protocol](docs/protocol.md): what the agent and a learning method may see, budgets, scoring
- [Methods](docs/methods.md): the three included methods and how to add your own
- [Data](docs/data.md): the dataset files and their fields

## Repository layout

```
servelearnbench/
  agent.py        the tool-calling agent loop
  llm.py          OpenAI-compatible client (streaming, retries, token accounting)
  runner.py       serving stream and window test sets for one method on one scenario
  methods/        Blind, Oracle, RAG, and the Method interface
  scenarios.py    the nine scenarios
  scoring.py      setting scores and averages
  protocol.py     budgets, the score line, the episode record, stream timestamps
  engine/         environment, tools, and the rule-based verifier
  domains/        retail/, banking/, pitch/ with tiers l1, l2, l3
tests/            offline tests (no API calls)
docs/             documentation and the leaderboard CSV
website/          the project page (GitHub Pages), built by website/build.py
examples/
```

## Citation

If you find ServeLearnBench useful, please cite:

```bibtex
@misc{zheng2026servelearnbench,
  title  = {ServeLearnBench: How Well Can Agents Self-Improve from Serving Experience?},
  author = {Zheng, Haizhong and Di, Yizhuo and Sadhukhan, Ranajoy and Jin, Shuowei and Chen, Beidi},
  year   = {2026}
}
```

## License

Apache 2.0. The retail domain's tool interface follows the design of [τ-bench](https://github.com/sierra-research/tau-bench).
