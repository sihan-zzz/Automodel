"""Debug: compare label masking between ChatDataset and StreamingPackedDataset."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from streaming_packed_dataset import StreamingPackedDataset

MODEL_PATH = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# --- ChatDataset (reference) ---
print("=" * 60)
print("ChatDataset (built-in, reference)")
print("=" * 60)
chat_ds = ChatDataset(
    path_or_dataset_id="allenai/tulu-3-sft-mixture",
    split="train[:10]",
    seq_length=8192,
    padding="do_not_pad",
    shuffle_seed=42,
    tokenizer=tok,
)

for i, sample in enumerate(chat_ds):
    if i >= 3:
        break
    ids = sample["input_ids"]
    labels = sample["labels"]
    n_total = len(ids)
    n_label = sum(1 for l in labels if l != -100)
    # Show token-by-token masking for first 30 tokens
    print(f"\nSample {i}: total={n_total} label={n_label} ({n_label/n_total:.0%})")
    for j in range(min(40, n_total)):
        tok_str = tok.decode([ids[j]])
        has_loss = "LOSS" if labels[j] != -100 else "mask"
        print(f"  [{j:3d}] id={ids[j]:6d} {has_loss} '{tok_str}'")
    if n_total > 40:
        print(f"  ... ({n_total - 40} more tokens)")

# --- StreamingPackedDataset ---
print("\n" + "=" * 60)
print("StreamingPackedDataset (our implementation)")
print("=" * 60)

# Load a real sample from Tulu3 to feed into our packer
DATA_PATH = "/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft"
with open(f"{DATA_PATH}/chunk.0.jsonl") as f:
    raw_sample = json.loads(f.readline())

# Simulate what _convert_and_tokenize does
dialog = raw_sample["dialog"]
keep_loss = raw_sample.get("keep_loss", None)

turns = []
roles = []
should_keep_loss = []
for i, turn in enumerate(dialog):
    role = turn.get("source", "user")
    content = turn.get("body", "")
    role_map = {"human": "user", "gpt": "assistant", "model": "assistant"}
    role = role_map.get(role, role.lower())
    if role in ("user", "assistant", "system") and content:
        turns.append(content)
        roles.append(role)
        if keep_loss and i < len(keep_loss):
            should_keep_loss.append(bool(keep_loss[i]))
        else:
            should_keep_loss.append(role == "assistant")

print(f"\nDialog: {len(turns)} turns, roles={roles}, keep_loss={should_keep_loss}")

# Build using our manual format
input_ids = []
labels = []

bos_ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
input_ids.extend(bos_ids)
labels.extend([-100] * len(bos_ids))

for i, (content, role, has_loss) in enumerate(zip(turns, roles, should_keep_loss)):
    gemma_role = "model" if role == "assistant" else role
    header = f"<|turn>{gemma_role}\n"
    footer = "<turn|>\n"

    header_ids = tok(header, add_special_tokens=False, return_tensors=None)["input_ids"]
    content_ids = tok(content, add_special_tokens=False, return_tensors=None)["input_ids"]
    footer_ids = tok(footer, add_special_tokens=False, return_tensors=None)["input_ids"]

    input_ids.extend(header_ids)
    labels.extend([-100] * len(header_ids))
    input_ids.extend(content_ids)
    labels.extend(list(content_ids) if has_loss else [-100] * len(content_ids))
    input_ids.extend(footer_ids)
    labels.extend([-100] * len(footer_ids))

n_total = len(input_ids)
n_label = sum(1 for l in labels if l != -100)
print(f"Our output: total={n_total} label={n_label} ({n_label/n_total:.0%})")

for j in range(min(40, n_total)):
    tok_str = tok.decode([input_ids[j]])
    has_loss_str = "LOSS" if labels[j] != -100 else "mask"
    print(f"  [{j:3d}] id={input_ids[j]:6d} {has_loss_str} '{tok_str}'")
if n_total > 40:
    print(f"  ... ({n_total - 40} more tokens)")

# Compare label ratios
print(f"\n{'=' * 60}")
print(f"ChatDataset label ratio: see above per-sample")
print(f"StreamingPacked label ratio: {n_label}/{n_total} = {n_label/n_total:.2%}")
