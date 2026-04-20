#!/bin/bash
#SBATCH --job-name=lf_vllm_fast
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/vllm_fast_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_fast
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks
LAUNCHER=$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher.py

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

mkdir -p $RESULTS_DIR

echo "=== vLLM Fast Eval (no enforce_eager, CUDA graphs enabled) ==="
echo "Model: $HF_MODEL"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

echo ''
echo 'Fast config: no enforce_eager, CUDA graphs + compile enabled'
echo 'DP=8, TP=1, max_model_len=32768'

# Key diff: NO enforce_eager — enables CUDA graphs + torch.compile for ~2-3x speedup
python3 $LAUNCHER --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768 \
    --tasks gpqa_diamond_thinking \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/gpqa_diamond_fast \
    --log_samples 2>&1

echo ''
echo '=== Done at \$(date) ==='
"
