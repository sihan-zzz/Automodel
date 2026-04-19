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

echo "=== Gemma4 26B-A4B-it Baseline Eval (lm-eval CLI) ==="

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
echo 'Running MMLU...'
lm_eval --model hf \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,trust_remote_code=True,attn_implementation=eager,parallelize=True \
    --tasks mmlu \
    --batch_size 4 \
    --limit 100 \
    --output_path $RESULTS_DIR/mmlu 2>&1 | tail -30

echo ''
echo 'Running IFEval...'
lm_eval --model hf \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,trust_remote_code=True,attn_implementation=eager,parallelize=True \
    --tasks ifeval \
    --batch_size 1 \
    --limit 50 \
    --output_path $RESULTS_DIR/ifeval 2>&1 | tail -30

echo ''
echo 'Running GPQA Diamond...'
lm_eval --model hf \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,trust_remote_code=True,attn_implementation=eager,parallelize=True \
    --tasks gpqa_diamond_zeroshot \
    --batch_size 4 \
    --limit 50 \
    --output_path $RESULTS_DIR/gpqa 2>&1 | tail -30

echo '=== All evals complete ==='
"
