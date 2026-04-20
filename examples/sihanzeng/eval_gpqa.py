#!/usr/bin/env python3
"""Direct vLLM GPQA eval — decodes with special tokens preserved.

Usage:
  python eval_gpqa.py --model /path/to/model --think
  python eval_gpqa.py --model /path/to/model --no-think
"""
import argparse
import json
import os
import re
import time

def load_gpqa(path):
    """Load GPQA from parquet or JSONL."""
    samples = []
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        for i in range(len(t)):
            samples.append({
                "question": t.column("question")[i].as_py(),
                "answer": t.column("answer")[i].as_py().strip().upper(),
            })
    elif path.endswith(".jsonl"):
        for line in open(path):
            d = json.loads(line)
            if "answer" in d:
                samples.append({"question": d["question"], "answer": d["answer"].strip().upper()})
            elif "label" in d:
                letters = ["A", "B", "C", "D"]
                choices = d["choices"]
                q = d["context"]
                if isinstance(choices, list):
                    q += "\n\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
                samples.append({"question": q, "answer": letters[d["label"]]})
    return samples


def extract_answer(text):
    """Extract answer letter from model output. Only searches the tail to avoid
    matching letters mentioned in mid-reasoning discussion."""
    # Only look at the last 500 chars for the answer
    tail = text[-500:]

    # Try structured patterns first
    m = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*\(?\s*([A-Da-d])\b", tail, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{([A-Da-d])\}", tail)
    if m:
        return m[-1].upper()
    # Last standalone letter in the tail
    m = re.findall(r"\b([A-D])\b", tail)
    if m:
        return m[-1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--no-think", dest="think", action="store_false")
    parser.add_argument("--max_tokens", type=int, default=31000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run_id", type=int, default=0, help="Run ID for variance tracking")
    parser.set_defaults(think=True)
    args = parser.parse_args()

    samples = load_gpqa(args.data)
    print(f"Loaded {len(samples)} samples")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    instruction = "Think step by step. After your reasoning, state your final answer as a single letter (A, B, C, or D) on its own line."
    prompts = []
    for s in samples:
        messages = [{"role": "user", "content": s["question"] + "\n\n" + instruction}]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.think
        ))

    mode = "think" if args.think else "nothink"
    print(f"Mode: {mode} | Run: {args.run_id}")
    print(f"Config: TP={args.tp} DP={args.dp} max_tokens={args.max_tokens}")

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
    )
    if args.dp > 1:
        llm_kwargs["data_parallel_size"] = args.dp
    llm = LLM(**llm_kwargs)

    seed = 42 + args.run_id * 1000
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_k=64, top_p=0.95, seed=seed)
    print(f"Seed: {seed}")

    print("Generating...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    # Score: use token IDs to find <channel|> (101), extract from clean text
    CHANNEL_CLOSE_ID = 101  # <channel|>
    correct = 0
    completed = 0
    results = []
    for i, (sample, out) in enumerate(zip(samples, outputs)):
        token_ids = list(out.outputs[0].token_ids)
        text_clean = out.outputs[0].text  # special tokens stripped

        # Check if <channel|> token exists in output
        has_channel = CHANNEL_CLOSE_ID in token_ids
        if has_channel:
            completed += 1
            # Find position of <channel|>, decode text BEFORE it (thinking part)
            ch_pos = len(token_ids) - 1 - token_ids[::-1].index(CHANNEL_CLOSE_ID)
            thinking_text = tok.decode(token_ids[:ch_pos], skip_special_tokens=True)
            extracted = extract_answer(thinking_text)
        else:
            # Truncated — fallback to last letter in clean text
            extracted = extract_answer(text_clean)

        target = sample["answer"]
        is_correct = extracted == target
        if is_correct:
            correct += 1

        results.append({
            "idx": i, "target": target, "extracted": extracted,
            "correct": is_correct, "has_channel": has_channel,
            "len_tokens": len(token_ids),
        })

    total = len(samples)
    completed_correct = sum(1 for r in results if r["has_channel"] and r["correct"])
    truncated = total - completed

    print()
    print("=" * 60)
    print(f"GPQA Diamond | {mode} | run={args.run_id}")
    print("=" * 60)
    print(f"Total:     {correct}/{total} = {correct*100/total:.1f}%")
    if completed > 0:
        print(f"Completed: {completed_correct}/{completed} = {completed_correct*100/completed:.1f}% ({completed}/{total} = {completed*100/total:.0f}% finished)")
    print(f"Truncated: {truncated}/{total} ({truncated*100/total:.0f}%)")
    print(f"Time:      {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
