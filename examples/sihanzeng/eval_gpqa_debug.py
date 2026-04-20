#!/usr/bin/env python3
"""Debug GPQA eval — saves full prompts and responses for inspection."""
import json
import os
import re
import time

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load data
    import pyarrow.parquet as pq
    t = pq.read_table(args.data)
    samples = []
    for i in range(min(args.n, len(t))):
        samples.append({
            "question": t.column("question")[i].as_py(),
            "answer": t.column("answer")[i].as_py().strip().upper(),
        })
    print("Loaded %d samples" % len(samples))

    # Build prompts
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    instruction = "Think step by step. After your reasoning, state your final answer as a single letter (A, B, C, or D) on its own line."
    prompts_think = []
    prompts_nothink = []
    for s in samples:
        messages = [{"role": "user", "content": s["question"] + "\n\n" + instruction}]
        prompts_think.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        prompts_nothink.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=args.max_model_len)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_k=64, top_p=0.95)

    results = []

    # Think mode
    print("\n=== Generating THINK mode ===")
    t0 = time.time()
    outputs_think = llm.generate(prompts_think, sp)
    think_time = time.time() - t0

    for i, (s, out) in enumerate(zip(samples, outputs_think)):
        text = out.outputs[0].text
        token_ids = list(out.outputs[0].token_ids)
        results.append({
            "idx": i,
            "mode": "think",
            "target": s["answer"],
            "prompt": prompts_think[i],
            "response": text,
            "response_token_ids_first_20": token_ids[:20],
            "response_token_ids_last_20": token_ids[-20:],
            "response_len_chars": len(text),
            "response_len_tokens": len(token_ids),
            "has_channel_text": "<channel|>" in text,
            "has_turn_text": "<turn|>" in text,
            "has_channel_token": any(tok.convert_ids_to_tokens([tid])[0] in ["<channel|>", "<|channel>"] for tid in token_ids[-50:]) if token_ids else False,
        })

    # No-think mode
    print("\n=== Generating NO-THINK mode ===")
    sp_short = SamplingParams(max_tokens=2048, temperature=1.0, top_k=64, top_p=0.95)
    t0 = time.time()
    outputs_nothink = llm.generate(prompts_nothink, sp_short)
    nothink_time = time.time() - t0

    for i, (s, out) in enumerate(zip(samples, outputs_nothink)):
        text = out.outputs[0].text
        token_ids = list(out.outputs[0].token_ids)
        results.append({
            "idx": i,
            "mode": "nothink",
            "target": s["answer"],
            "prompt": prompts_nothink[i],
            "response": text,
            "response_token_ids_first_20": token_ids[:20],
            "response_token_ids_last_20": token_ids[-20:],
            "response_len_chars": len(text),
            "response_len_tokens": len(token_ids),
        })

    # Decode special token IDs
    print("\n=== Special token check ===")
    for name in ["<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>"]:
        ids = tok.encode(name, add_special_tokens=False)
        print("  %s -> token_ids=%s" % (name, ids))

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved %d results to %s" % (len(results), args.output))
    print("Think: %.0fs, No-think: %.0fs" % (think_time, nothink_time))

if __name__ == "__main__":
    main()
