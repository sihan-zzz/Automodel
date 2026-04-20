"""Debug script: print first N batches from StreamingPackedDataset to check for duplicates."""
import os
import sys
import hashlib
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import AutoTokenizer
from streaming_packed_dataset import StreamingPackedDataset

MODEL_PATH = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

ds = StreamingPackedDataset(
    sources=[
        {"path": "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/step_3p5_flash_sft", "weight": 0.5},
        {"path": "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/jackrong_sft", "weight": 0.5},
    ],
    tokenizer=tok,
    seq_length=16384,
    packed_sequence_size=16384,
    shuffle_buffer_size=64,
    seed=42,
)

seen_hashes = {}
n_samples = 200
n_dupes = 0

print(f"Checking first {n_samples} packed samples for duplicates...\n")

for i, sample in enumerate(ds):
    if i >= n_samples:
        break

    ids = sample["input_ids"]
    labels = sample["labels"]
    n_label = sum(1 for l in labels if l != -100)
    n_total = len(ids)

    # Hash the input_ids to detect duplicates
    h = hashlib.md5(str(ids[:100]).encode()).hexdigest()[:12]

    # Decode first 50 tokens to see content
    first_tokens = tok.decode(ids[:50], skip_special_tokens=False)

    if h in seen_hashes:
        n_dupes += 1
        print(f"  DUPLICATE: sample {i} matches sample {seen_hashes[h]}")
        print(f"    hash={h} label_ratio={n_label/n_total:.2f}")
    else:
        seen_hashes[h] = i

    if i < 10 or i % 50 == 0:
        print(f"sample {i:3d}: hash={h} tokens={n_total} labels={n_label} ({n_label/n_total:.0%})")
        print(f"  preview: {first_tokens[:120]}...")

print(f"\nTotal: {min(i+1, n_samples)} samples, {n_dupes} duplicates, {len(seen_hashes)} unique")
print(f"Drop stats: {ds._drop_count} dropped / {ds._total_count} total")
