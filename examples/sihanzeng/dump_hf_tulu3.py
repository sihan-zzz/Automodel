"""Dump first 1000 HF Tulu3 samples to local JSONL for fair comparison."""
import json
from datasets import load_dataset

ds = load_dataset("allenai/tulu-3-sft-mixture", split="train[:1000]")
out = "/tmp/tulu3_1k.jsonl"
with open(out, "w") as f:
    for sample in ds:
        # Convert HF format (messages) to amaia format (dialog) for streaming reader
        dialog = []
        keep_loss = []
        for msg in sample["messages"]:
            dialog.append({"source": msg["role"], "body": msg["content"]})
            keep_loss.append(msg["role"] == "assistant")
        f.write(json.dumps({"dialog": dialog, "keep_loss": keep_loss, "messages": sample["messages"]}) + "\n")
print(f"Wrote {len(ds)} samples to {out}")
