#!/bin/bash
#SBATCH --job-name=gpqa_native
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/bench_%A_%a.log
#SBATCH --array=0-3

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_bench
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# 4 benchmarks in parallel, one per node
declare -a TASKS LABELS LIMITS
TASKS[0]="gpqa_diamond_native_think"
TASKS[1]="mmlu_pro"
TASKS[2]="bbh_cot_zeroshot"
TASKS[3]="aime_2026"
LABELS[0]="gpqa_diamond"
LABELS[1]="mmlu_pro"
LABELS[2]="bbh"
LABELS[3]="aime_2026"
# Limits: 0 = full dataset. Set >0 for subsampling
LIMITS[0]=0
LIMITS[1]=500
LIMITS[2]=500
LIMITS[3]=0

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
LIMIT=${LIMITS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== Benchmark: $LABEL ==="
echo "Task: $TASK"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache /tmp/triton_cache
export HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache TRITON_CACHE_DIR=/tmp/triton_cache

pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

echo 'vLLM:' \$(python3 -c 'import vllm; print(vllm.__version__)' 2>&1)
echo 'lm-eval:' \$(python3 -c 'import lm_eval; print(lm_eval.__version__)' 2>&1)
echo ''
echo 'Starting: $TASK'
echo 'Gen start: '\$(date)

LAUNCHER=\$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher_minimal.py
python3 \$LAUNCHER --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768,enable_thinking=True,think_end_token='<channel|>' \
    --tasks $TASK \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    \$([ $LIMIT -gt 0 ] && echo "--limit $LIMIT") \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at '\$(date)' ==='
"
