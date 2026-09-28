#!/usr/bin/env bash
# Evaluate a model served locally by vLLM or SGLang (OpenAI-compatible, tool calling enabled).
#   vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 8000
set -euo pipefail
slb run --scenario banking_l1 --method rag \
  --model Qwen/Qwen3-32B --base-url http://localhost:8000/v1 --reasoning-effort none \
  --judge-model accounts/fireworks/models/glm-5p3-flash
