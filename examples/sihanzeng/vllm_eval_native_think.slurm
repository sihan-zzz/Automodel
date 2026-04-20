#!/bin/bash
#SBATCH --job-name=lf_bench
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

# Custom tasks use our thinking_flex filter (answer BEFORE <channel|>)
# BBH uses built-in extractors with think_end_token=NOMATCH (full output to extractors)
declare -a TASKS LABELS LIMITS THINK_TOKENS
TASKS[0]="gpqa_diamond_native_think"
TASKS[1]="mmlu_pro_think"
TASKS[2]="bbh_cot_zeroshot"
TASKS[3]="aime_2026"
LABELS[0]="gpqa_diamond"
LABELS[1]="mmlu_pro"
LABELS[2]="bbh"
LABELS[3]="aime_2026"
LIMITS[0]=0
LIMITS[1]=500
LIMITS[2]=500
LIMITS[3]=0
# BBH uses NOMATCH so built-in extractors get full thinking output
THINK_TOKENS[0]="<channel|>"
THINK_TOKENS[1]="<channel|>"
THINK_TOKENS[2]="NOMATCH_NEVER"
THINK_TOKENS[3]="<channel|>"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
LIMIT=${LIMITS[$SLURM_ARRAY_TASK_ID]}
THINK_TOKEN=${THINK_TOKENS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== Benchmark: $LABEL ==="
echo "Task: $TASK | Limit: $LIMIT | think_end_token: $THINK_TOKEN"
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
LOCAL_MODEL=/tmp/model_local
cp -r $HF_MODEL \$LOCAL_MODEL
echo 'Done copying'

pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

echo 'vLLM:' \$(python3 -c 'import vllm; print(vllm.__version__)' 2>&1)
echo 'Starting: $TASK'
echo 'Gen start: '\$(date)

LAUNCHER=\$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher_minimal.py
LIMIT_FLAG=''
if [ $LIMIT -gt 0 ]; then LIMIT_FLAG='--limit $LIMIT'; fi

python3 \$LAUNCHER --model vllm \
    --model_args pretrained=\$LOCAL_MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768,enable_thinking=True,think_end_token='$THINK_TOKEN' \
    --tasks $TASK \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples \$LIMIT_FLAG 2>&1

echo ''
echo '=== Done: $LABEL at '\$(date)' ==='
"
