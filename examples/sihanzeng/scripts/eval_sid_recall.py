"""Evaluate SID recall: description → SID token generation.

Loads sid_eval JSONL, generates completions via vLLM, compares
predicted SID tokens against ground truth per-position.

Usage:
  python eval_sid_recall.py --model /path/to/model --data /path/to/sid_eval_3_16.jsonl

Metrics:
  format_ok: fraction with valid <TOKEN>...<ad_5/X></TOKEN> format
  pos1..pos6: per-position exact match (independent)
  exact: all 6 positions match
"""
import argparse
import json
import re
import sys


def parse_sid(text: str) -> list[str] | None:
    """Extract 6-token SID from text. Returns list of 6 tokens or None."""
    m = re.search(r"<TOKEN>((?:<ad_\d+/\d+>){6})</TOKEN>", text)
    if not m:
        return None
    tokens = re.findall(r"<ad_(\d+/\d+)>", m.group(1))
    return tokens if len(tokens) == 6 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/eval/sid_eval_3_16.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    samples = [json.loads(line) for line in open(args.data)]
    print(f"Loaded {len(samples)} samples from {args.data}")

    prompts = [s["prompt"] for s in samples]
    targets = [parse_sid(s["text"]) for s in samples]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["</TOKEN>", "\n"],
    )

    outputs = llm.generate(prompts, params)

    format_ok = 0
    pos_match = [0] * 6
    exact_match = 0
    results = []

    for i, (out, target) in enumerate(zip(outputs, targets)):
        gen_text = out.outputs[0].text
        full_text = prompts[i] + gen_text
        pred = parse_sid(full_text)

        is_format_ok = pred is not None
        if is_format_ok:
            format_ok += 1

        per_pos = [False] * 6
        if pred and target:
            all_match = True
            for p in range(6):
                if pred[p] == target[p]:
                    pos_match[p] += 1
                    per_pos[p] = True
                else:
                    all_match = False
            if all_match:
                exact_match += 1

        results.append({
            "prompt": prompts[i][-80:],
            "target": target,
            "pred": pred,
            "format_ok": is_format_ok,
            "pos_match": per_pos,
        })

    n = len(samples)
    metrics = {
        "format_ok": format_ok / n,
        "exact_match": exact_match / n,
        **{f"pos{p+1}": pos_match[p] / n for p in range(6)},
        "n": n,
    }

    print("\n=== SID Recall Results ===")
    print(f"Model: {args.model}")
    print(f"Samples: {n}")
    print(f"Format OK: {metrics['format_ok']:.1%}")
    print(f"Exact match: {metrics['exact_match']:.1%}")
    for p in range(6):
        print(f"  pos{p+1}: {metrics[f'pos{p+1}']:.1%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
