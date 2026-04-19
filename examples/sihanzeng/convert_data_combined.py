"""Convert and mix Tulu3 + Step 3.5 data to OpenAI messages format for NeMo Automodel."""
import json
import os
import random
from pathlib import Path

TULU3_DIR = "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft"
STEP35_DIR = "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/step_3p5_flash_sft"
OUTPUT_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/tulu3_step35_combined"

TULU3_WEIGHT = 0.8
STEP35_WEIGHT = 0.2
SEED = 42

def convert_tulu3_sample(sample):
    messages = []
    for turn in sample.get("dialog", []):
        messages.append({
            "role": turn.get("source", "user"),
            "content": turn.get("body", ""),
        })
    return {"messages": messages} if messages else None

def convert_step35_sample(sample):
    messages = []
    for turn in sample.get("dialog", sample.get("conversations", [])):
        if isinstance(turn, dict):
            role = turn.get("source", turn.get("role", turn.get("from", "user")))
            content = turn.get("body", turn.get("content", turn.get("value", "")))
            if role.lower() in ("human", "user"):
                role = "user"
            elif role.lower() in ("gpt", "assistant", "model"):
                role = "assistant"
            elif role.lower() == "system":
                role = "system"
            messages.append({"role": role, "content": content})
    return {"messages": messages} if messages else None

def load_jsonl_dir(data_dir, converter, max_files=None):
    samples = []
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.jsonl"))
    if max_files:
        files = files[:max_files]
    for f in files:
        with open(f) as fin:
            for line in fin:
                try:
                    sample = json.loads(line.strip())
                    converted = converter(sample)
                    if converted and len(converted["messages"]) >= 2:
                        samples.append(converted)
                except:
                    continue
    return samples

print(f"Loading Tulu3 from {TULU3_DIR}...")
tulu3_samples = load_jsonl_dir(TULU3_DIR, convert_tulu3_sample)
print(f"  Loaded {len(tulu3_samples)} Tulu3 samples")

print(f"Loading Step 3.5 from {STEP35_DIR}...")
step35_samples = load_jsonl_dir(STEP35_DIR, convert_step35_sample)
print(f"  Loaded {len(step35_samples)} Step 3.5 samples")

# Mix with weights
random.seed(SEED)
n_tulu3 = int(len(tulu3_samples) * TULU3_WEIGHT)
n_step35 = int(len(step35_samples) * STEP35_WEIGHT) if step35_samples else 0

# If Step 3.5 is smaller, adjust ratio
total_target = n_tulu3 + n_step35
if total_target == 0:
    total_target = len(tulu3_samples)
    n_tulu3 = total_target
    n_step35 = 0

sampled_tulu3 = random.sample(tulu3_samples, min(n_tulu3, len(tulu3_samples)))
sampled_step35 = random.sample(step35_samples, min(n_step35, len(step35_samples))) if step35_samples else []

combined = sampled_tulu3 + sampled_step35
random.shuffle(combined)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "train.jsonl")
with open(output_file, 'w') as fout:
    for sample in combined:
        fout.write(json.dumps(sample) + "\n")

print(f"\nCombined dataset:")
print(f"  Tulu3: {len(sampled_tulu3)} ({len(sampled_tulu3)/len(combined)*100:.1f}%)")
print(f"  Step 3.5: {len(sampled_step35)} ({len(sampled_step35)/len(combined)*100:.1f}%)")
print(f"  Total: {len(combined)}")
print(f"  Output: {output_file}")
