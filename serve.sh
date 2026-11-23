#!/usr/bin/env bash
# Starts vLLM with every adapter in ./adapters, the way we ran it.
#   ./serve.sh            8 adapter slots, max rank 64 (what we run in production)
#   SLOTS=16 RANK=32 ./serve.sh   16 slots, max rank 32 (run download.sh with MAX_RANK=32 first)
set -euo pipefail
cd "$(dirname "$0")"
SLOTS=${SLOTS:-8}
RANK=${RANK:-64}
MODULES=()
while read -r name; do MODULES+=("$name=adapters/$name"); done < models.txt

exec vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype bfloat16 --port 8000 \
  --enable-lora --max-lora-rank "$RANK" --max-loras "$SLOTS" --max-cpu-loras 32 \
  --lora-modules "${MODULES[@]}" \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3
