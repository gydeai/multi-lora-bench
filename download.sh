#!/usr/bin/env bash
# Downloads the adapters in adapters.json at the exact revisions we tested, into ./adapters/<name>,
# and writes models.txt (one served name per line) for bench.py.
# Needs: pip install -U huggingface_hub   (and python3)
# MAX_RANK=32 skips adapters above rank 32 (for the 16-slot setup).
set -euo pipefail
cd "$(dirname "$0")"
MAX_RANK=${MAX_RANK:-64}
: > models.txt
python3 -c '
import json, sys
for a in json.load(open("adapters.json")):
    if a["loaded_8_slots_rank64"] and a["rank"] <= int(sys.argv[1]):
        print(a["repo"], a["revision"])
' "$MAX_RANK" | while read -r repo rev; do
  name=$(echo "$repo" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
  hf download "$repo" adapter_config.json adapter_model.safetensors --revision "$rev" --local-dir "adapters/$name" >/dev/null
  echo "$name" >> models.txt
  echo "downloaded $repo"
done
echo "$(wc -l < models.txt) adapters in models.txt"
