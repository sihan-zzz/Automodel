#!/bin/bash
#SBATCH --job-name=gpqa_eval
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/gpqa_eval_%A_%a.log
#SBATCH --array=0-1

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
DATA=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/eval/gpqa_diamond/test/gpqa_diamond.parquet
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gpqa_direct
SCRIPT=$HOME/Automodel/examples/sihanzeng/eval_gpqa.py

# 0: thinking mode, 1: no-think mode
declare -a THINK_FLAGS LABELS
THINK_FLAGS[0]="--think"
THINK_FLAGS[1]="--no-think --max_tokens 2048 --max_model_len 4096"
LABELS[0]="think"
LABELS[1]="nothink"

FLAG=${THINK_FLAGS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== GPQA Diamond Direct Eval: $LABEL ==="
echo "Model: $HF_MODEL"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
export HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache TRITON_CACHE_DIR=/tmp/triton_cache PYTHONUNBUFFERED=1

pip install pyarrow 2>&1 | tail -1

python3 $SCRIPT \
    --model $HF_MODEL \
    --data $DATA \
    --dp 8 --tp 1 \
    --gpu_util 0.90 \
    $FLAG \
    --output $RESULTS_DIR/${LABEL}.jsonl 2>&1
"
