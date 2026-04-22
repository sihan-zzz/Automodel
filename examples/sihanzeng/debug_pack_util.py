"""Compare packing utilization: offline vs online."""
import sys
sys.path.insert(0, "examples/sihanzeng")

import torch
from transformers import AutoTokenizer
from streaming_gemma4_dataset import StreamingGemma4Dataset
from nemo_automodel.components.datasets.utils import packed_thd_collater_with_mm
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from torch.utils.data import DataLoader

MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def analyze(name, dl, n=10):
    print(f"\n=== {name} ===")
    print(f"{'batch':>5} | {'total':>6} | {'real':>6} {'%':>4} | {'label':>6} {'%':>4} | {'masked':>6} {'%':>4} | {'pad':>6} {'%':>4}")
    for i, batch in enumerate(dl):
        if i >= n:
            break
        ids = batch["input_ids"]
        labels = batch["labels"]
        total = ids.numel()
        n_label = (labels != -100).sum().item()
        n_pad = (ids == 0).sum().item()
        n_real = total - n_pad
        n_masked = n_real - n_label
        print(f"{i:5d} | {total:6d} | {n_real:6d} {n_real/total:4.0%} | {n_label:6d} {n_label/total:4.0%} | {n_masked:6d} {n_masked/total:4.0%} | {n_pad:6d} {n_pad/total:4.0%}")

# Offline
print("Loading ChatDataset...")
chat_ds = ChatDataset(
    path_or_dataset_id="allenai/tulu-3-sft-mixture", split="train[:5000]",
    seq_length=4096, padding="do_not_pad", truncation=True, shuffle_seed=42, tokenizer=tok,
)
print("Packing offline...")
packed = pack_dataset(chat_ds, split="train", packed_sequence_size=4096, padding_idx=0)
dl1 = DataLoader(packed, batch_size=2, collate_fn=packed_thd_collater_with_mm, shuffle=False)
analyze("Offline (ChatDS + pack_dataset)", dl1)

# Online
print("\nLoading StreamingGemma4Dataset...")
stream_ds = StreamingGemma4Dataset(
    sources=[{"path": "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft", "weight": 1.0}],
    tokenizer=tok, seq_length=4096, pack_size=4096, shuffle_buffer_size=64,
    convert_thinking=False, seed=42,
)
dl2 = DataLoader(stream_ds, batch_size=2, collate_fn=packed_thd_collater_with_mm)
analyze("Online (StreamingGemma4Dataset + pack_size)", dl2)
