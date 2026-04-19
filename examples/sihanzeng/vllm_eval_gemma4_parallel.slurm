#!/bin/bash
#SBATCH --job-name=lf_hf_eval
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/hf_eval_%A_%a.log
#SBATCH --array=0-3

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_gemma4_dp8
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks

export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# Split benchmarks across 4 nodes
# Local datasets: mmlu_local, gpqa_local (from Lustre)
# HuggingFace datasets: gsm8k_cot_zeroshot, hendrycks_math, ifeval, bbh_cot_zeroshot
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

echo "=== HF Gemma4 26B-A4B-it Eval (DP=8) ==="
echo "Group: $LABEL | Tasks: $TASK"
echo "Model: $HF_MODEL"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

pip install --target=/tmp/lm_eval_pkg lm-eval 2>&1 | tail -5
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

python3 -c 'import transformers; print(f\"transformers {transformers.__version__}\")' 2>&1
python3 -c 'import lm_eval; print(f\"lm-eval {lm_eval.__version__}\")' 2>&1
python3 -c 'import accelerate; print(f\"accelerate {accelerate.__version__}\")' 2>&1

echo ''
echo 'Starting: $TASK (data parallel, 8 GPUs)'

# Data parallelism: 8 full model replicas, one per GPU
# Each GPU: 52GB model + 131GB free for KV cache
# ~8x throughput vs model parallelism
accelerate launch --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,trust_remote_code=True,attn_implementation=eager,max_length=8192 \
    --tasks $TASK \
    --batch_size 16 \
    --apply_chat_template \
    --num_fewshot 0 \
    --gen_kwargs do_sample=True,temperature=1.0,top_k=64,top_p=1.0 \
    --include_path $TASK_DIR \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at \$(date) ==='
"
