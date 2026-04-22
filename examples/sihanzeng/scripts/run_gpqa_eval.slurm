#!/bin/bash
# Usage: MODEL_PATH=/path/to/model EVAL_NAME=gpqa_step100 sbatch run_gpqa_eval.slurm
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/%x_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:?Must set MODEL_PATH}
EVAL_NAME=${EVAL_NAME:-gpqa_eval}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_${EVAL_NAME}
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks
LAUNCHER=$HOME/Automodel/examples/sihanzeng/scripts/vllm_eval_launcher_minimal.py
DP=${DP_SIZE:-4}

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

mkdir -p $RESULTS_DIR

echo "=== GPQA Eval: $EVAL_NAME ==="
echo "Model: $HF_MODEL"
echo "DP=$DP | Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

python3 $LAUNCHER --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=$DP,gpu_memory_utilization=0.90,max_model_len=32768,enable_thinking=True,think_end_token='<channel|>' \
    --tasks gpqa_diamond_native_think \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR \
    --log_samples 2>&1

echo 'Done at '\$(date)
"
