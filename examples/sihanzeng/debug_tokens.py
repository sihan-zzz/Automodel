"""Dump token IDs and label IDs from both data loaders for comparison."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from streaming_packed_dataset import StreamingPackedDataset, packed_thd_collater_with_mm
from nemo_automodel.components.datasets.utils import default_collater_with_mm

MODEL_PATH = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

IGNORE = -100

def dump_batch(name, batch, tok, batch_idx):
    ids = batch["input_ids"]
    labels = batch["labels"]
    print(f"\n{'='*70}")
    print(f"{name} — batch {batch_idx}")
    print(f"{'='*70}")
    if isinstance(ids, torch.Tensor):
        # batched
        for s in range(ids.shape[0]):
            sample_ids = ids[s].tolist()
            sample_labels = labels[s].tolist()
            n_total = len(sample_ids)
            n_label = sum(1 for l in sample_labels if l != IGNORE)
            n_masked = n_total - n_label
            print(f"\n  Sample {s}: total={n_total} label={n_label} masked={n_masked} ratio={n_label/n_total:.1%}")
            # Print first 60 tokens with labels
            for j in range(min(60, n_total)):
                tok_str = tok.decode([sample_ids[j]]).replace('\n', '\\n')
                lbl = sample_labels[j]
                tag = "LOSS" if lbl != IGNORE else "mask"
                print(f"    [{j:4d}] id={sample_ids[j]:6d} label={lbl:6d} {tag}  '{tok_str}'")
            # Find the transition point where labels change from mask to LOSS
            transitions = []
            for j in range(1, min(200, n_total)):
                prev = "mask" if sample_labels[j-1] == IGNORE else "LOSS"
                curr = "mask" if sample_labels[j] == IGNORE else "LOSS"
                if prev != curr:
                    tok_str = tok.decode([sample_ids[j]]).replace('\n', '\\n')
                    transitions.append(f"[{j}] {prev}->{curr} id={sample_ids[j]} '{tok_str}'")
            if transitions:
                print(f"    Transitions (first 200 tokens):")
                for t in transitions:
                    print(f"      {t}")
    else:
        # single sample (list)
        n_total = len(ids)
        n_label = sum(1 for l in labels if l != IGNORE)
        print(f"  total={n_total} label={n_label} ratio={n_label/n_total:.1%}")


# === ChatDataset ===
print("\n" + "#"*70)
print("# ChatDataset (HF Tulu3, built-in masking)")
print("#"*70)

chat_ds = ChatDataset(
    path_or_dataset_id="allenai/tulu-3-sft-mixture",
    split="train[:20]",
    seq_length=8192,
    padding="do_not_pad",
    shuffle_seed=42,
    tokenizer=tok,
)

from torch.utils.data import DataLoader
chat_dl = DataLoader(chat_ds, batch_size=2, collate_fn=default_collater_with_mm, shuffle=False)

for i, batch in enumerate(chat_dl):
    if i >= 3:
        break
    dump_batch("ChatDataset", batch, tok, i)


# === StreamingPackedDataset ===
print("\n" + "#"*70)
print("# StreamingPackedDataset (our implementation)")
print("#"*70)

stream_ds = StreamingPackedDataset(
    sources=[
        {"path": "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/step_3p5_flash_sft", "weight": 0.5},
        {"path": "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/jackrong_sft", "weight": 0.5},
    ],
    tokenizer=tok,
    seq_length=16384,
    packed_sequence_size=16384,
    shuffle_buffer_size=0,  # no shuffle for deterministic output
    seed=42,
)

stream_dl = DataLoader(stream_ds, batch_size=1, collate_fn=packed_thd_collater_with_mm)

for i, batch in enumerate(stream_dl):
    if i >= 3:
        break
    dump_batch("StreamingPacked", batch, tok, i)
