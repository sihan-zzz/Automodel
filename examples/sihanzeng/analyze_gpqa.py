import json, re, glob, sys

base = sys.argv[1] if len(sys.argv) > 1 else "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_fast"
files = sorted(glob.glob(f"{base}/**/*.jsonl", recursive=True))
if not files:
    print("No files found in", base)
    sys.exit(1)

print("File:", files[-1])
total = ch_ok = ch_correct = starts_thought = no_think = 0
for i, line in enumerate(open(files[-1])):
    d = json.loads(line)
    output = d["resps"][0][0] if isinstance(d["resps"][0], list) else d["resps"][0]
    target = d["target"]
    total += 1
    if output.startswith("thought"):
        starts_thought += 1
    elif not output.startswith("<"):
        no_think += 1
    if "<channel|>" in output:
        ch_ok += 1
        after = output.split("<channel|>")[-1].strip()
        letter = re.search(r"^([A-Da-d])", after)
        if letter and letter.group(1).upper() == target:
            ch_correct += 1
    if i < 5:
        has_ch = "<channel|>" in output
        print(f"  #{i}: target={target} think={output[:15]!r}... has_ch={has_ch} last30={output[-30:]!r}")

print(f"\nTotal: {total}")
print(f"Starts with 'thought': {starts_thought}")
print(f"No thinking pattern: {no_think}")
print(f"Has <channel|>: {ch_ok} ({ch_ok*100//max(total,1)}%)")
if ch_ok > 0:
    print(f"Correct (channel extract): {ch_correct}/{ch_ok} = {ch_correct*100/ch_ok:.1f}%")
