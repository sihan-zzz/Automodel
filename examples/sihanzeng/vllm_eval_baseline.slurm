#!/bin/bash
#SBATCH --job-name=hf_eval
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/hf_eval_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
HF_MODEL=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_baseline

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR $RESULTS_DIR

echo "=== Gemma4 26B-A4B-it Baseline Eval (HF backend) ==="
echo "Model: $HF_MODEL"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     python3 -c "
import torch, json, os, time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

model_path = '$HF_MODEL'
results_dir = '$RESULTS_DIR'

print('Loading model...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'Model loaded in {time.time()-t0:.1f}s')

# Simple MMLU eval (5-shot, multiple choice)
print('\\n=== MMLU (5-shot, log-prob) ===')
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4, max_length=4096)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=['mmlu'],
        batch_size=4,
        limit=100,  # Quick test: 100 samples per subtask
    )
    mmlu_acc = results['results'].get('mmlu', {}).get('acc,none', 'N/A')
    print(f'MMLU accuracy: {mmlu_acc}')
    with open(os.path.join(results_dir, 'mmlu_results.json'), 'w') as f:
        json.dump(results['results'], f, indent=2, default=str)
except Exception as e:
    print(f'MMLU eval failed: {e}')

# IFEval
print('\\n=== IFEval ===')
try:
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=['ifeval'],
        batch_size=1,
        limit=50,
    )
    ifeval_acc = results['results'].get('ifeval', {})
    print(f'IFEval results: {ifeval_acc}')
    with open(os.path.join(results_dir, 'ifeval_results.json'), 'w') as f:
        json.dump(results['results'], f, indent=2, default=str)
except Exception as e:
    print(f'IFEval eval failed: {e}')

# GPQA Diamond
print('\\n=== GPQA Diamond ===')
try:
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=['gpqa_diamond_zeroshot'],
        batch_size=4,
        limit=50,
    )
    gpqa_acc = results['results'].get('gpqa_diamond_zeroshot', {})
    print(f'GPQA Diamond results: {gpqa_acc}')
    with open(os.path.join(results_dir, 'gpqa_results.json'), 'w') as f:
        json.dump(results['results'], f, indent=2, default=str)
except Exception as e:
    print(f'GPQA eval failed: {e}')

print('\\n=== All evals complete ===')
"
