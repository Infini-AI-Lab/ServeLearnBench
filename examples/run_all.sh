#!/usr/bin/env bash
# Run Blind, Oracle and RAG on all nine scenarios for one model, then score.
#   MODEL=accounts/fireworks/models/glm-5p3-flash bash examples/run_all.sh
set -euo pipefail
: "${MODEL:?set MODEL}"
SCENARIOS=retail_l1,retail_l2,retail_l3,banking_l1,banking_l2,banking_l3,pitch_l1,pitch_l2,pitch_l3
for METHOD in blind oracle rag; do
  slb run --scenario "$SCENARIOS" --method "$METHOD" --model "$MODEL" --workers "${WORKERS:-8}" --resume
done
slb score results/"${MODEL##*/}"/*.jsonl
