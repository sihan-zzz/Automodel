#!/bin/bash
# Evaluate S0 warmup variants: partial freeze + untied
# Array: 0=partial_199, 1=untied_199, 2=untied_399, 3=untied_599
#SBATCH --job-name=nemo_eval_s0v
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/eval_s0v_%A_%a.log
#SBATCH --array=0-3

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/vllm-v0.19.1-cu130.sqsh
EVAL_DATA=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/data/eval/sid_eval_3_16.jsonl
RESULTS_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/eval_s0_variants
SCRIPTS_DIR=$HOME/Automodel/examples/sihanzeng/scripts
TASK_DIR=$HOME/Automodel/examples/sihanzeng/eval_tasks

declare -a MODELS LABELS
MODELS[0]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/gemma4_cpt_s0_partial/epoch_0_step_199/model/consolidated
MODELS[1]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/gemma4_cpt_s0_untied/epoch_0_step_199/model/consolidated
MODELS[2]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/gemma4_cpt_s0_untied/epoch_0_step_399/model/consolidated
MODELS[3]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_runs/gemma4_cpt_s0_untied/epoch_0_step_599/model/consolidated
LABELS[0]=partial_199
LABELS[1]=untied_199
LABELS[2]=untied_399
LABELS[3]=untied_599

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
RESULTS=$RESULTS_BASE/$LABEL

export PYTHONUNBUFFERED=1
mkdir -p $RESULTS

echo "=== Eval: $LABEL ==="
echo "Model: $MODEL"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-mount-home \
     bash -c "
mkdir -p /tmp/hf_cache /tmp/xdg_cache
export HF_HOME=/tmp/hf_cache XDG_CACHE_HOME=/tmp/xdg_cache PYTHONUNBUFFERED=1

echo '=== SID Eval ==='
python3 $SCRIPTS_DIR/eval_sid_recall.py \
    --model $MODEL \
    --data $EVAL_DATA \
    --output $RESULTS/sid_eval.json \
    --tp 1 2>&1

echo ''
echo '=== GPQA Diamond ==='
pip install --target=/tmp/lm_eval_pkg lm-eval ray 2>&1 | tail -1
export PYTHONPATH=/tmp/lm_eval_pkg:\$PYTHONPATH

python3 $SCRIPTS_DIR/vllm_eval_launcher_minimal.py --model vllm \
    --model_args pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.90,max_model_len=32768,enable_thinking=True,think_end_token='<channel|>' \
    --tasks gpqa_diamond_native_think \
    --batch_size 64 \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --include_path $TASK_DIR \
    --output_path $RESULTS/gpqa \
    --log_samples 2>&1

echo '=== Done $LABEL at '\$(date)' ==='
"
