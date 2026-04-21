"""Direct comparison: our _parse_dialog output vs ChatDataset format."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from streaming_gemma4_dataset import _parse_dialog, tokenize_with_loss_mask
from transformers import AutoTokenizer

MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
IGNORE = -100

# Load a local Tulu3 sample
with open("/mnt/lustre/metavmds0lstre/teams/ads_cpt/datasets/tulu3_sft/chunk.0.jsonl") as f:
    sample = json.loads(f.readline())

# Our pipeline
parsed = _parse_dialog(sample)
messages, loss_flags = parsed
print("=== Our _parse_dialog output ===")
for i, (m, l) in enumerate(zip(messages, loss_flags)):
    print(f"  [{i}] role={m['role']} loss={l} content={m['content'][:60]}...")

# Tokenize with our pipeline
result = tokenize_with_loss_mask(tok, messages, loss_flags, 8192)
if result:
    our_ids, our_labels = result
    n_label = sum(1 for l in our_labels if l != IGNORE)
    print(f"\nOur: {len(our_ids)} input_ids, {n_label} labels")
    print(f"  First 10 ids: {our_ids[:10]}")
    print(f"  First 10 labels: {our_labels[:10]}")
else:
    print("Our tokenization returned None!")

# ChatDataset-style: same messages, apply_chat_template directly
tokenized = tok.apply_chat_template(messages, tokenize=True, return_dict=True)
chat_ids = list(tokenized["input_ids"])
# ChatDataset shifts: input_ids[:-1], labels = original[1:]
chat_input = chat_ids[:-1]
chat_labels_raw = list(chat_ids)
# Build mask using same loss_flags
# (simplified — just check if our ids match)
print(f"\nChatDataset-style: {len(chat_input)} input_ids")
print(f"  First 10 ids: {chat_input[:10]}")

# Compare
print(f"\n=== Comparison ===")
print(f"  Our len: {len(our_ids)}, Chat len: {len(chat_input)}")
if our_ids == chat_input:
    print("  input_ids: MATCH!")
else:
    print("  input_ids: MISMATCH!")
    for j in range(min(len(our_ids), len(chat_input))):
        if our_ids[j] != chat_input[j]:
            print(f"  First diff at [{j}]: ours={our_ids[j]} ({tok.decode([our_ids[j]])!r}) vs chat={chat_input[j]} ({tok.decode([chat_input[j]])!r})")
            break
    if len(our_ids) != len(chat_input):
        print(f"  Length diff: {len(our_ids)} vs {len(chat_input)}")
