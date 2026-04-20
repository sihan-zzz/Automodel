#!/bin/bash
#SBATCH --job-name=lf_vllm_speed
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/vllm_speed_%A_%a.log
#SBATCH --array=0-2

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_speed
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks
LAUNCHER=$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher.py

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# Speed test configs
# 0: sampling + DP=8 TP=1 (baseline with improved prompt, max_gen_toks=31000)
# 1: sampling + TP=2 DP=4 (shard model across 2 GPUs, less mem-bw bottleneck)
# 2: sampling + DP=8 TP=1 + speculative decoding disabled + max_num_batched_tokens=32768
declare -a VLLM_ARGS LABELS
VLLM_ARGS[0]="tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768"
VLLM_ARGS[1]="tensor_parallel_size=2,data_parallel_size=4,gpu_memory_utilization=0.90,max_model_len=32768"
VLLM_ARGS[2]="tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.95,max_model_len=32768,max_num_batched_tokens=32768"
LABELS[0]="dp8_tp1"
LABELS[1]="dp4_tp2"
LABELS[2]="dp8_tp1_highutil"

VLLM_ARG=${VLLM_ARGS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== Speed Test: $LABEL ==="
echo "vLLM args: $VLLM_ARG"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache

pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

echo 'Config: $LABEL'
echo 'Start generation: '\$(date)

python3 $LAUNCHER --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,$VLLM_ARG \
    --tasks gpqa_diamond_thinking_sample \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at '\$(date)' ==='
"
