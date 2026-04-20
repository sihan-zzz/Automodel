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
#SBATCH --array=0-5

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
DATA=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/eval/gpqa_diamond/test/gpqa_diamond.parquet
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gpqa_v2
SCRIPT=$HOME/Automodel/examples/sihanzeng/eval_gpqa.py

# 0-2: thinking mode (3 runs for variance), 3-5: no-think mode (3 runs)
if [ $SLURM_ARRAY_TASK_ID -lt 3 ]; then
    MODE="--think"
    LABEL="think"
    RUN_ID=$SLURM_ARRAY_TASK_ID
    MAX_TOKENS="--max_tokens 31000 --max_model_len 32768"
else
    MODE="--no-think"
    LABEL="nothink"
    RUN_ID=$((SLURM_ARRAY_TASK_ID - 3))
    MAX_TOKENS="--max_tokens 2048 --max_model_len 4096"
fi

mkdir -p $RESULTS_DIR

echo "=== GPQA Diamond: $LABEL run=$RUN_ID ==="
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
export HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache TRITON_CACHE_DIR=/tmp/triton_cache PYTHONUNBUFFERED=1

pip install pyarrow -q 2>&1 | tail -1

python3 $SCRIPT \
    --model $HF_MODEL \
    --data $DATA \
    --tp 1 --dp 1 \
    --gpu_util 0.90 \
    --run_id $RUN_ID \
    $MODE $MAX_TOKENS \
    --output $RESULTS_DIR/${LABEL}_run${RUN_ID}.jsonl 2>&1
"
