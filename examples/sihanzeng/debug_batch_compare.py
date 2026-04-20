"""Compare exact batch tensors: ChatDataset vs StreamingPackedDataset on same data."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from streaming_packed_dataset import StreamingPackedDataset
from nemo_automodel.components.datasets.utils import default_collater_with_mm, packed_sequence_thd_collater

MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
IGNORE = -100

print("="*60)
print("ChatDataset")
print("="*60)
chat_ds = ChatDataset(
    path_or_dataset_id="allenai/tulu-3-sft-mixture",
    split="train[:5]",
    seq_length=8192,
    padding="do_not_pad",
    shuffle_seed=42,
    tokenizer=tok,
)
from torch.utils.data import DataLoader
chat_dl = DataLoader(chat_ds, batch_size=1, collate_fn=default_collater_with_mm, shuffle=False)

for i, batch in enumerate(chat_dl):
    if i >= 2: break
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    n = len(ids)
    n_label = sum(1 for l in labels if l != IGNORE)

    print(f"\nSample {i}: {n} tokens, {n_label} labels ({n_label/n:.0%})")

    # Check: are labels shifted? labels[i] should be ids[i+1]
    match_next = 0
    match_same = 0
    for j in range(n-1):
        if labels[j] == IGNORE: continue
        if labels[j] == ids[j+1]: match_next += 1
        if labels[j] == ids[j]: match_same += 1
    print(f"  labels[i]==ids[i+1]: {match_next}/{n_label} ({match_next/max(n_label,1):.0%})")
    print(f"  labels[i]==ids[i]:   {match_same}/{n_label} ({match_same/max(n_label,1):.0%})")

    # Show first transition
    for j in range(1, min(200, n)):
        if labels[j-1] == IGNORE and labels[j] != IGNORE:
            print(f"  First LOSS at [{j}]: id={ids[j]} label={labels[j]} next_id={ids[j+1] if j+1<n else 'END'}")
            print(f"    ids[{j-2}:{j+3}] = {ids[max(0,j-2):j+3]}")
            print(f"    labels[{j-2}:{j+3}] = {labels[max(0,j-2):j+3]}")
            break

print("\n" + "="*60)
print("StreamingPackedDataset")
print("="*60)

stream_ds = StreamingPackedDataset(
    sources=[{"path": "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft", "weight": 1.0}],
    tokenizer=tok,
    seq_length=8192,
    packed_sequence_size=8192,
    shuffle_buffer_size=0,
    seed=42,
)

from streaming_packed_dataset import packed_thd_collater_with_mm
stream_dl = DataLoader(stream_ds, batch_size=1, collate_fn=packed_thd_collater_with_mm)

for i, batch in enumerate(stream_dl):
    if i >= 2: break
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    n = len(ids)
    n_label = sum(1 for l in labels if l != IGNORE)

    print(f"\nPacked sample {i}: {n} tokens, {n_label} labels ({n_label/n:.0%})")

    match_next = 0
    match_same = 0
    for j in range(n-1):
        if labels[j] == IGNORE: continue
        if labels[j] == ids[j+1]: match_next += 1
        if labels[j] == ids[j]: match_same += 1
    print(f"  labels[i]==ids[i+1]: {match_next}/{n_label} ({match_next/max(n_label,1):.0%})")
    print(f"  labels[i]==ids[i]:   {match_same}/{n_label} ({match_same/max(n_label,1):.0%})")

    # Check seq_lens
    if "seq_lens" in batch:
        sl = batch["seq_lens"][0].tolist()
        sl = [s for s in sl if s != -1000]
        print(f"  seq_lens: {sl[:5]}... ({len(sl)} seqs)")

    # Show first transition
    for j in range(1, min(200, n)):
        if labels[j-1] == IGNORE and labels[j] != IGNORE:
            print(f"  First LOSS at [{j}]: id={ids[j]} label={labels[j]} next_id={ids[j+1] if j+1<n else 'END'}")
            print(f"    ids[{j-2}:{j+3}] = {ids[max(0,j-2):j+3]}")
            print(f"    labels[{j-2}:{j+3}] = {labels[max(0,j-2):j+3]}")
            break

    # Show a window around position 37 (where content typically starts)
    for j in [35, 36, 37, 38, 39, 40]:
        if j < n:
            tok_str = tok.decode([ids[j]]).replace('\n','\\n')
            print(f"    [{j}] id={ids[j]:6d} label={labels[j]:6d} {'LOSS' if labels[j]!=IGNORE else 'mask'} '{tok_str}'")
