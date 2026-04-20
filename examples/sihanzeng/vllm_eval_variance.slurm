#!/bin/bash
#SBATCH --job-name=lf_var
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/var_%A_%a.log
#SBATCH --array=0-5

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_variance
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# 0-2: maj@3 (repeats=3, majority_vote) with seeds 42/1042/2042
# 3-5: single pass (repeats=1) with seeds 42/1042/2042
if [ $SLURM_ARRAY_TASK_ID -lt 3 ]; then
    TASK="gpqa_diamond_native_think"
    MODE="maj3"
    RUN_ID=$SLURM_ARRAY_TASK_ID
else
    TASK="gpqa_diamond_single"
    MODE="single"
    RUN_ID=$((SLURM_ARRAY_TASK_ID - 3))
fi
SEED=$((42 + RUN_ID * 1000))
LABEL="${MODE}_seed${SEED}"

mkdir -p $RESULTS_DIR

echo "=== Variance Test: $LABEL ==="
echo "Task: $TASK | Mode: $MODE | Seed: $SEED"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
export HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache TRITON_CACHE_DIR=/tmp/triton_cache

echo 'Copying model to local SSD...'
cp -r $HF_MODEL /tmp/model_local
echo 'Done'

pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

echo 'Mode: $MODE | Seed: $SEED'
echo 'Gen start: '\$(date)

LAUNCHER=\$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher_minimal.py
python3 \$LAUNCHER --model vllm \
    --model_args pretrained=/tmp/model_local,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768,enable_thinking=True,think_end_token='<channel|>' \
    --tasks $TASK \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --seed $SEED \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at '\$(date)' ==='
"
