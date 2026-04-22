"""Dump HF Tulu3 samples to local JSONL for fair comparison.

Usage:
  python dump_hf_tulu3.py          # default 5000 samples
  python dump_hf_tulu3.py 10000    # custom count
"""
import json
import os
import sys
from datasets import load_dataset

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
ds = load_dataset("allenai/tulu-3-sft-mixture", split=f"train[:{n}]")
out_dir = f"/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/tulu3_{n // 1000}k"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "train.jsonl")
with open(out, "w") as f:
    for sample in ds:
        dialog = []
        keep_loss = []
        for msg in sample["messages"]:
            dialog.append({"source": msg["role"], "body": msg["content"]})
            keep_loss.append(msg["role"] == "assistant")
        f.write(json.dumps({"dialog": dialog, "keep_loss": keep_loss, "messages": sample["messages"]}) + "\n")
print(f"Wrote {len(ds)} samples to {out}")
