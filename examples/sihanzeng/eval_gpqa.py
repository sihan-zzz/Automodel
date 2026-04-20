#!/usr/bin/env python3
"""Direct vLLM GPQA eval — no lm-eval dependency.

Usage:
  python eval_gpqa.py --model /path/to/model --think --split diamond
  python eval_gpqa.py --model /path/to/model --no-think --split diamond
"""
import argparse
import json
import os
import re
import time

def load_gpqa(path, split="diamond"):
    """Load GPQA from parquet or JSONL."""
    if os.path.isdir(path):
        # Try parquet first, then jsonl
        for sub in [f"gpqa_{split}/test/gpqa_{split}.parquet", f"gpqa_{split}.parquet",
                     f"gpqa_{split}.jsonl", f"domains/gpqa_{split}.jsonl", f"gpqa_main.jsonl"]:
            full = os.path.join(path, sub)
            if os.path.exists(full):
                path = full
                break

    samples = []
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        for i in range(len(t)):
            q = t.column("question")[i].as_py()
            a = t.column("answer")[i].as_py().strip().upper()
            samples.append({"question": q, "answer": a})
    elif path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
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

    print(f"Loaded {len(samples)} samples from {path}")
    return samples


def extract_answer(output):
    """Extract last answer letter (A-D) from model output, looking BEFORE <channel|>."""
    if "<channel|>" in output:
        text = output.split("<channel|>")[0]
    else:
        text = output

    # Try structured patterns first
    m = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*\(?\s*([A-Da-d])\b", text, re.IGNORECASE)
    if m:
        return m[-1].upper()

    # Try boxed
    m = re.findall(r"\\boxed\{([A-Da-d])\}", text)
    if m:
        return m[-1].upper()

    # Last standalone letter
    m = re.findall(r"\b([A-D])\b", text)
    if m:
        return m[-1]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--data", default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/eval/gpqa_diamond/test/gpqa_diamond.parquet")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("--no-think", dest="think", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (overrides temp)")
    parser.add_argument("--max_tokens", type=int, default=31000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--output", default=None, help="Save results JSONL")
    parser.set_defaults(think=True)
    args = parser.parse_args()

    samples = load_gpqa(args.data)

    # Build prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    instruction = "Think step by step. After your reasoning, state your final answer as a single letter (A, B, C, or D) on its own line."

    prompts = []
    for s in samples:
        messages = [{"role": "user", "content": f"{s['question']}\n\n{instruction}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.think
        )
        prompts.append(prompt)

    print(f"Mode: {'thinking' if args.think else 'no-think'}")
    print(f"Config: TP={args.tp} DP={args.dp} max_tokens={args.max_tokens} max_model_len={args.max_model_len}")
    if args.greedy:
        print("Decoding: greedy (temperature=0)")
    else:
        print(f"Decoding: sampling (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p})")
    print(f"Prompt[0] (first 200 chars): {prompts[0][:200]}")
    print()

    # Load vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
    )

    if args.greedy:
        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    else:
        sp = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    print("Generating...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    # Score
    correct = 0
    total = len(samples)
    results = []
    for i, (sample, out) in enumerate(zip(samples, outputs)):
        text = out.outputs[0].text
        target = sample["answer"]
        extracted = extract_answer(text)
        is_correct = extracted == target
        if is_correct:
            correct += 1

        has_channel = "<channel|>" in text
        result = {
            "idx": i,
            "target": target,
            "extracted": extracted,
            "correct": is_correct,
            "has_channel": has_channel,
            "output_len": len(text),
            "output_tail": text[-100:] if len(text) > 100 else text,
        }
        results.append(result)

    # Stats
    completed = sum(1 for r in results if r["has_channel"])
    completed_correct = sum(1 for r in results if r["has_channel"] and r["correct"])
    truncated = total - completed

    print()
    print("=" * 60)
    print(f"GPQA Diamond Results ({'thinking' if args.think else 'no-think'})")
    print("=" * 60)
    print(f"Total:     {correct}/{total} = {correct*100/total:.1f}%")
    print(f"Completed: {completed_correct}/{completed} = {completed_correct*100/max(completed,1):.1f}% ({completed}/{total} finished thinking)")
    print(f"Truncated: {truncated}/{total} ({truncated*100/total:.0f}%)")
    print(f"Time:      {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"Tokens/s:  {sum(r['output_len'] for r in results)/elapsed:.0f} output toks/s total")
    print("=" * 60)

    # Show some errors
    errors = [r for r in results if not r["correct"]][:3]
    if errors:
        print(f"\nFirst {len(errors)} errors:")
        for r in errors:
            print(f"  #{r['idx']}: target={r['target']} extracted={r['extracted']} channel={r['has_channel']} ...{r['output_tail'][-60:]}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
