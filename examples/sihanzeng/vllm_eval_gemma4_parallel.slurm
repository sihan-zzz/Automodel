#!/bin/bash
#SBATCH --job-name=lf_vllm_eval
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/vllm_eval_%A_%a.log
#SBATCH --array=0-3

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
HF_MODEL=${MODEL_PATH:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it}
RESULTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_vllm_gemma4

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1

# Split benchmarks across 4 nodes for parallel eval
# Matches Google Gemma IT eval setup: all 0-shot, sampling-based
declare -a TASKS LABELS
TASKS[0]="mmlu"
TASKS[1]="mmlu_pro,gpqa_diamond_zeroshot"
TASKS[2]="gsm8k_cot_zeroshot,hendrycks_math"
TASKS[3]="ifeval,bbh_cot_zeroshot"
LABELS[0]="mmlu"
LABELS[1]="mmlu_pro_gpqa"
LABELS[2]="gsm8k_math"
LABELS[3]="ifeval_bbh"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $RESULTS_DIR

echo "=== vLLM Gemma4 26B-A4B-it Eval ==="
echo "Group: $LABEL | Tasks: $TASK"
echo "Model: $HF_MODEL"
echo "Node: $(hostname) | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED \
     --container-mount-home \
     bash -c "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Keep container's transformers 5.5.0 (needed for gemma4 model type), install lm-eval only
pip install --target=/tmp/lm_eval_pkg lm-eval 2>&1 | tail -5
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

python3 -c 'import transformers; print(f\"transformers {transformers.__version__}\")' 2>&1
python3 -c 'import lm_eval; print(f\"lm-eval {lm_eval.__version__}\")' 2>&1

echo ''
echo 'Starting: $TASK'
# Use launcher that patches transformers/vLLM dataclass incompatibility
python3 ~/Automodel/examples/sihanzeng/vllm_eval_launcher.py --model vllm \
    --model_args pretrained=$HF_MODEL,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.95,max_model_len=8192,enforce_eager=True,trust_remote_code=True \
    --tasks $TASK \
    --batch_size auto \
    --apply_chat_template \
    --num_fewshot 0 \
    --gen_kwargs temperature=1.0,top_k=64,top_p=1.0 \
    --output_path $RESULTS_DIR/$LABEL \
    --log_samples 2>&1

echo ''
echo '=== Done: $LABEL at \$(date) ==='
"
