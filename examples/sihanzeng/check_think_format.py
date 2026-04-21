"""Check Gemma4 native thinking format vs <think> tags."""
from transformers import AutoTokenizer

MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Test 1: Native thinking via reasoning_content
msgs_native = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4.", "reasoning_content": "Let me think... 2+2=4"},
]
out1 = tok.apply_chat_template(msgs_native, tokenize=False)
print("=== Native thinking (reasoning_content) ===")
print(out1)
print()

# Test 2: <think> tags in content (Step3.5/JackRong format)
msgs_tag = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>Let me think... 2+2=4</think>The answer is 4."},
]
out2 = tok.apply_chat_template(msgs_tag, tokenize=False)
print("=== <think> tags in content ===")
print(out2)
print()

# Test 3: Check thinking-related special tokens
print("=== Thinking-related tokens ===")
vocab = tok.get_vocab()
for pattern in ['think', 'channel', 'thought', '|think']:
    matches = [(t, i) for t, i in vocab.items() if pattern in t.lower()]
    if matches:
        print(f"  '{pattern}': {matches[:5]}")
