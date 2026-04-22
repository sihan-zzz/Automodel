"""Study embedding state in partial-freeze checkpoint vs base model."""
import json
import os
import torch
from safetensors import safe_open

ORIG_VOCAB = 262144
BASE_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it-expanded-tied"
CKPT_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/gemma4_cpt_s0_partial/epoch_0_step_199/model/consolidated"
EMBED_KEY = "model.language_model.embed_tokens.weight"


def load_embed(directory):
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(directory, f), framework="pt") as sf:
                if EMBED_KEY in sf.keys():
                    return sf.get_tensor(EMBED_KEY)
    return None


print("=== 1. Config check ===")
cfg = json.load(open(os.path.join(CKPT_DIR, "config.json")))
print(f"  tie_word_embeddings (top): {cfg.get('tie_word_embeddings')}")
tc = cfg.get("text_config", {})
print(f"  tie_word_embeddings (text_config): {tc.get('tie_word_embeddings')}")

idx = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))
has_lm = any("lm_head" in k for k in idx["weight_map"])
print(f"  separate lm_head in safetensors: {has_lm}")

print("\n=== 2. Load embeddings ===")
base_embed = load_embed(BASE_DIR)
ckpt_embed = load_embed(CKPT_DIR)
print(f"  base: {base_embed.shape}, ckpt: {ckpt_embed.shape}")

print("\n=== 3. Original token embeddings intact? ===")
orig_diff = (ckpt_embed[:ORIG_VOCAB] - base_embed[:ORIG_VOCAB]).abs()
print(f"  max diff:  {orig_diff.max().item():.10f}")
print(f"  mean diff: {orig_diff.mean().item():.10f}")
changed = (orig_diff.sum(dim=1) > 1e-6).sum().item()
print(f"  rows with any change: {changed} / {ORIG_VOCAB}")

print("\n=== 4. New SID token embeddings changed? ===")
new_diff = (ckpt_embed[ORIG_VOCAB:] - base_embed[ORIG_VOCAB:]).abs()
print(f"  max diff:  {new_diff.max().item():.6f}")
print(f"  mean diff: {new_diff.mean().item():.6f}")
n_new = ckpt_embed.shape[0] - ORIG_VOCAB
changed_new = (new_diff.sum(dim=1) > 1e-6).sum().item()
print(f"  rows changed: {changed_new} / {n_new}")

print("\n=== 5. Embedding magnitudes (L2 norm per row) ===")
orig_norms = ckpt_embed[:ORIG_VOCAB].float().norm(dim=1)
new_norms = ckpt_embed[ORIG_VOCAB:].float().norm(dim=1)
print(f"  original tokens: mean={orig_norms.mean():.4f}  std={orig_norms.std():.4f}  min={orig_norms.min():.4f}  max={orig_norms.max():.4f}")
print(f"  new SID tokens:  mean={new_norms.mean():.4f}  std={new_norms.std():.4f}  min={new_norms.min():.4f}  max={new_norms.max():.4f}")

# Gemma4 scales embeddings by sqrt(hidden_size)
hidden_size = 2816
scale = hidden_size ** 0.5
print(f"\n  Gemma4 embed scale factor: sqrt({hidden_size}) = {scale:.1f}")
print(f"  Effective original logit scale: {orig_norms.mean() * scale:.1f}")
print(f"  Effective new token logit scale: {new_norms.mean() * scale:.1f}")
print(f"  Ratio new/original: {new_norms.mean() / orig_norms.mean():.3f}")
