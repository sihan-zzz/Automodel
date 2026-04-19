"""Convert Tulu3 and Step 3.5 separately to OpenAI messages JSONL for ChatDataset."""
import json, os
from pathlib import Path

TULU3_DIR = "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft"
STEP35_DIR = "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/step_3p5_flash_sft"
OUT_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/sft_datasets"

def convert_dialog(sample):
    messages = []
    for turn in sample.get("dialog", sample.get("conversations", [])):
        if isinstance(turn, dict):
            role = turn.get("source", turn.get("role", turn.get("from", "user")))
            content = turn.get("body", turn.get("content", turn.get("value", "")))
            role_map = {"human": "user", "gpt": "assistant", "model": "assistant"}
            role = role_map.get(role.lower(), role.lower())
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})
    return messages

def process_dir(data_dir, output_file):
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.jsonl"))
    count = 0
    with open(output_file, 'w') as fout:
        for f in files:
            with open(f) as fin:
                for line in fin:
                    try:
                        sample = json.loads(line.strip())
                        messages = convert_dialog(sample)
                        if len(messages) >= 2:
                            fout.write(json.dumps({"messages": messages}) + "\n")
                            count += 1
                    except:
                        continue
    print(f"  {count} samples -> {output_file}")
    return count

os.makedirs(OUT_DIR, exist_ok=True)

print("Converting Tulu3...")
n1 = process_dir(TULU3_DIR, os.path.join(OUT_DIR, "tulu3.jsonl"))

print("Converting Step 3.5...")
n2 = process_dir(STEP35_DIR, os.path.join(OUT_DIR, "step35.jsonl"))

print(f"\nDone: {n1} Tulu3 + {n2} Step 3.5 = {n1+n2} total")
print(f"Files: {OUT_DIR}/tulu3.jsonl, {OUT_DIR}/step35.jsonl")
