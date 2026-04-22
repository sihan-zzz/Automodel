"""Expand Gemma4 tokenizer and model embeddings with SID tokens.

Adds 1,538 new special tokens to Gemma4:
  - <TOKEN>, </TOKEN>  (SID delimiters)
  - <ad_0/0> .. <ad_5/255>  (6 positions × 256 codebook values)

Expands embed_tokens and lm_head, sets tie_word_embeddings=False,
initializes new embeddings with mean of existing embeddings (scaled appropriately).

Usage:
  python expand_tokenizer_and_embeddings.py \
    --model_path /path/to/gemma-4-26B-A4B-it \
    --output_path /path/to/gemma-4-26B-A4B-it-expanded \
    --num_positions 6 --codebook_size 256
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
    logger.info(f"Adding {len(sid_tokens)} SID tokens ({args.num_positions} positions × {args.codebook_size} codebook + 2 delimiters)")

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    orig_vocab_size = len(tokenizer)
    logger.info(f"Original vocab size: {orig_vocab_size}")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens})
    new_vocab_size = len(tokenizer)
    logger.info(f"Added {num_added} tokens, new vocab size: {new_vocab_size}")

    token_id = tokenizer.convert_tokens_to_ids("<TOKEN>")
    end_token_id = tokenizer.convert_tokens_to_ids("</TOKEN>")
    first_codebook_id = tokenizer.convert_tokens_to_ids("<ad_0/0>")
    last_codebook_id = tokenizer.convert_tokens_to_ids(f"<ad_{args.num_positions-1}/{args.codebook_size-1}>")
    logger.info(f"<TOKEN> = {token_id}, </TOKEN> = {end_token_id}")
    logger.info(f"Codebook range: {first_codebook_id} .. {last_codebook_id}")

    dtype = getattr(torch, args.torch_dtype)
    logger.info(f"Loading model from {args.model_path} (dtype={args.torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True,
    )

    old_embed = model.get_input_embeddings()
    logger.info(f"Original embedding shape: {old_embed.weight.shape}")

    model.resize_token_embeddings(new_vocab_size)

    new_embed = model.get_input_embeddings()
    logger.info(f"New embedding shape: {new_embed.weight.shape}")

    # Initialize new token embeddings with mean of existing embeddings
    with torch.no_grad():
        mean_embed = old_embed.weight[:orig_vocab_size].mean(dim=0)
        std_embed = old_embed.weight[:orig_vocab_size].std()
        new_embed.weight[orig_vocab_size:] = mean_embed.unsqueeze(0) + torch.randn(
            num_added, mean_embed.shape[0], dtype=dtype, device=mean_embed.device,
        ) * std_embed * 0.02
        logger.info(f"Initialized {num_added} new embeddings (mean + 0.02*std noise)")

        # If lm_head exists and is separate, initialize it too
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != new_embed.weight.data_ptr():
            mean_head = lm_head.weight[:orig_vocab_size].mean(dim=0)
            std_head = lm_head.weight[:orig_vocab_size].std()
            lm_head.weight[orig_vocab_size:] = mean_head.unsqueeze(0) + torch.randn(
                num_added, mean_head.shape[0], dtype=dtype, device=mean_head.device,
            ) * std_head * 0.02
            logger.info(f"Initialized lm_head for {num_added} new tokens")

    # Untie embeddings
    if model.config.tie_word_embeddings:
        logger.info("Untying word embeddings (setting tie_word_embeddings=False)")
        model.config.tie_word_embeddings = False
        # After resize_token_embeddings with tied weights, lm_head.weight IS embed_tokens.weight
        # We need to create a separate lm_head with its own copy
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() == new_embed.weight.data_ptr():
            import torch.nn as nn
            new_lm_head = nn.Linear(new_embed.weight.shape[1], new_vocab_size, bias=False)
            new_lm_head.weight = nn.Parameter(new_embed.weight.clone())
            model.set_output_embeddings(new_lm_head)
            logger.info("Created separate lm_head (cloned from embed_tokens)")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving expanded model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Copy processor_config.json if it exists (needed by vLLM for Gemma4)
    proc_config = Path(args.model_path) / "processor_config.json"
    if proc_config.exists():
        shutil.copy2(proc_config, output_path / "processor_config.json")
        logger.info("Copied processor_config.json")

    # Verify
    logger.info("--- Verification ---")
    logger.info(f"Config vocab_size: {model.config.vocab_size}")
    logger.info(f"Config tie_word_embeddings: {model.config.tie_word_embeddings}")
    logger.info(f"Embed shape: {model.get_input_embeddings().weight.shape}")
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        logger.info(f"LM head shape: {lm_head.weight.shape}")
        logger.info(f"Weights tied: {lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")

    test_tokens = ["<TOKEN>", "</TOKEN>", "<ad_0/0>", "<ad_2/128>", f"<ad_{args.num_positions-1}/{args.codebook_size-1}>"]
    for tok in test_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        back = tokenizer.convert_ids_to_tokens(tid)
        logger.info(f"  {tok} -> id={tid} -> '{back}'")


if __name__ == "__main__":
    main()
