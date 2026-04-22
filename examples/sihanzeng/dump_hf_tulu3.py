"""Dump first 1000 HF Tulu3 samples to local JSONL for fair comparison."""
import json
import os
from datasets import load_dataset

ds = load_dataset("allenai/tulu-3-sft-mixture", split="train[:1000]")
out_dir = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/tulu3_1k"
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
