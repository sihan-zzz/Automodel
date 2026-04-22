#!/bin/bash
#SBATCH --job-name=nemo_expand_tok
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/expand_tok_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
SCRIPT=$HOME/Automodel/examples/sihanzeng/scripts/expand_tokenizer_and_embeddings.py
MODEL=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it
OUTPUT=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it-expanded

export PYTHONUNBUFFERED=1

echo "=== Expanding Gemma4 tokenizer with SID tokens ==="
echo "Input:  $MODEL"
echo "Output: $OUTPUT"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-mount-home \
     bash -c "
python3 $SCRIPT \
    --model_path $MODEL \
    --output_path $OUTPUT \
    --num_positions 6 \
    --codebook_size 256 \
    --torch_dtype bfloat16
"
