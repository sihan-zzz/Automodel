#!/bin/bash
#SBATCH --job-name=save_fsdp2
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/save_fsdp2_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
SCRIPT=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/scripts/save_fsdp2_ckpt.py

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR

echo "=== Saving Gemma4-IT as FSDP2 checkpoint ==="
srun --ntasks=1 --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED,CUDA_DEVICE_MAX_CONNECTIONS \
     --container-mount-home \
     bash -c "
pip install torchdata -q 2>&1 | tail -1
torchrun --nproc-per-node=8 $SCRIPT
"
