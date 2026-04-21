"""Compare exact batch output: ChatDataset vs StreamingGemma4Dataset on same Tulu3."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from streaming_gemma4_dataset import StreamingGemma4Dataset
from nemo_automodel.components.datasets.utils import default_collater_with_mm
from torch.utils.data import DataLoader

MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
IGNORE = -100

# --- ChatDataset ---
print("=" * 60)
print("ChatDataset (HF Tulu3)")
print("=" * 60)
chat_ds = ChatDataset(
    path_or_dataset_id="allenai/tulu-3-sft-mixture",
    split="train[:20]",
    seq_length=8192,
    padding="do_not_pad",
    truncation=True,
    shuffle_seed=42,
    tokenizer=tok,
)

chat_dl = DataLoader(chat_ds, batch_size=2, collate_fn=default_collater_with_mm, shuffle=False)

for i, batch in enumerate(chat_dl):
    if i >= 3:
        break
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    n = len(ids)
    n_label = sum(1 for l in labels if l != IGNORE)

    # Verify shift
    match_next = sum(1 for j in range(n-1) if labels[j] != IGNORE and labels[j] == ids[j+1])

    print(f"\nBatch {i}, sample 0: {n} tokens, {n_label} labels, shift_match={match_next}/{n_label}")
    # First few tokens
    for j in range(min(8, n)):
        t = tok.decode([ids[j]]).replace('\n', '\\n')
        print(f"  [{j:3d}] id={ids[j]:6d} label={labels[j]:6d} '{t}'")
    # First loss transition
    for j in range(1, min(200, n)):
        if labels[j-1] == IGNORE and labels[j] != IGNORE:
            for k in range(max(0, j-2), min(j+5, n)):
                t = tok.decode([ids[k]]).replace('\n', '\\n')
                tag = "LOSS" if labels[k] != IGNORE else "mask"
                print(f"  [{k:3d}] id={ids[k]:6d} label={labels[k]:6d} {tag} '{t}'")
            break

# --- StreamingGemma4Dataset ---
print("\n" + "=" * 60)
print("StreamingGemma4Dataset (local Tulu3)")
print("=" * 60)

stream_ds = StreamingGemma4Dataset(
    sources=[{"path": "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft", "weight": 1.0}],
    tokenizer=tok,
    seq_length=8192,
    shuffle_buffer_size=0,
    convert_thinking=False,
    seed=42,
)

stream_dl = DataLoader(stream_ds, batch_size=2, collate_fn=default_collater_with_mm)

for i, batch in enumerate(stream_dl):
    if i >= 3:
        break
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    n = len(ids)
    n_label = sum(1 for l in labels if l != IGNORE)

    match_next = sum(1 for j in range(n-1) if labels[j] != IGNORE and labels[j] == ids[j+1])

    print(f"\nBatch {i}, sample 0: {n} tokens, {n_label} labels, shift_match={match_next}/{n_label}")
    for j in range(min(8, n)):
        t = tok.decode([ids[j]]).replace('\n', '\\n')
        print(f"  [{j:3d}] id={ids[j]:6d} label={labels[j]:6d} '{t}'")
    for j in range(1, min(200, n)):
        if labels[j-1] == IGNORE and labels[j] != IGNORE:
            for k in range(max(0, j-2), min(j+5, n)):
                t = tok.decode([ids[k]]).replace('\n', '\\n')
                tag = "LOSS" if labels[k] != IGNORE else "mask"
                print(f"  [{k:3d}] id={ids[k]:6d} label={labels[k]:6d} {tag} '{t}'")
            break
