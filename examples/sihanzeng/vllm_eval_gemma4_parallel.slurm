#!/bin/bash
#SBATCH --job-name=lf_vllm_eval
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/vllm_eval_%A_%a.log
#SBATCH --array=0-3

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_vllm
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks
LAUNCHER=$HOME/Automodel/examples/sihanzeng/vllm_eval_launcher.py

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# Split benchmarks across 4 nodes
declare -a TASKS LABELS
TASKS[0]="mmlu_local"
TASKS[1]="gpqa_local"
TASKS[2]="gsm8k_cot_zeroshot,hendrycks_math"
TASKS[3]="ifeval,bbh_cot_zeroshot"
LABELS[0]="mmlu"
LABELS[1]="gpqa"
LABELS[2]="gsm8k_math"
LABELS[3]="ifeval_bbh"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== vLLM 0.19.1 Gemma4 26B-A4B-it Eval (TP=8) ==="
echo "Group: $LABEL | Tasks: $TASK"
echo "Model: $HF_MODEL"
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

pip install --target=/tmp/lm_eval_pkg lm-eval ray langdetect 2>&1 | tail -3
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

python3 -c 'import vllm; print(f\"vLLM {vllm.__version__}\")' 2>&1
python3 -c 'import lm_eval; print(f\"lm-eval {lm_eval.__version__}\")' 2>&1

echo ''
echo 'Starting: $TASK (vLLM TP=8)'
python3 $LAUNCHER --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=8192,enforce_eager=True \
    --tasks $TASK \
    --batch_size 256 \
    --apply_chat_template \
    --num_fewshot 0 \
    --gen_kwargs temperature=1.0,top_k=64,top_p=1.0 \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at \$(date) ==='
"
