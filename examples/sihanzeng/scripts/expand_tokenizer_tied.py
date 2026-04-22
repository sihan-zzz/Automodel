"""Expand Gemma4 tokenizer and model embeddings with SID tokens (tied version).

Same as expand_tokenizer_and_embeddings.py but keeps tie_word_embeddings=True.
The shared embedding/lm_head tensor is expanded together.

Usage:
  python expand_tokenizer_tied.py \
    --model_path /path/to/gemma-4-26B-A4B-it \
    --output_path /path/to/gemma-4-26B-A4B-it-expanded-tied
"""
import argparse
import logging
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_sid_tokens(num_positions: int = 6, codebook_size: int = 256) -> list[str]:
    tokens = ["<TOKEN>", "</TOKEN>"]
    for pos in range(num_positions):
        for idx in range(codebook_size):
            tokens.append(f"<ad_{pos}/{idx}>")
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_positions", type=int, default=6)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    sid_tokens = build_sid_tokens(args.num_positions, args.codebook_size)
    logger.info(f"Adding {len(sid_tokens)} SID tokens (keeping tie_word_embeddings=True)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    orig_vocab_size = len(tokenizer)
    logger.info(f"Original vocab size: {orig_vocab_size}")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens})
    new_vocab_size = len(tokenizer)
    logger.info(f"Added {num_added} tokens, new vocab size: {new_vocab_size}")

    dtype = getattr(torch, args.torch_dtype)
    logger.info(f"Loading model (dtype={args.torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True,
    )

    logger.info(f"tie_word_embeddings: {model.config.tie_word_embeddings}")
    old_embed = model.get_input_embeddings()
    logger.info(f"Original embedding shape: {old_embed.weight.shape}")

    model.resize_token_embeddings(new_vocab_size)

    new_embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = lm_head is not None and lm_head.weight.data_ptr() == new_embed.weight.data_ptr()
    logger.info(f"New embedding shape: {new_embed.weight.shape}, tied: {tied}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    proc_config = Path(args.model_path) / "processor_config.json"
    if proc_config.exists():
        shutil.copy2(proc_config, output_path / "processor_config.json")

    token_id = tokenizer.convert_tokens_to_ids("<TOKEN>")
    logger.info(f"<TOKEN> = {token_id}, </TOKEN> = {tokenizer.convert_tokens_to_ids('</TOKEN>')}")
    logger.info(f"Codebook: {tokenizer.convert_tokens_to_ids('<ad_0/0>')} .. {tokenizer.convert_tokens_to_ids(f'<ad_{args.num_positions-1}/{args.codebook_size-1}>')}")
    logger.info(f"Config tie_word_embeddings: {model.config.tie_word_embeddings}")


if __name__ == "__main__":
    main()
