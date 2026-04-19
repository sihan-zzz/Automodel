#!/bin/bash
#SBATCH --job-name=nemo_am_blend
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/nemo_am_blend_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel
SCRIPTS_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/scripts
CONFIG=$AUTOMODEL_DIR/examples/sihanzeng/gemma4_26b_sft_packed.yaml

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$AUTOMODEL_DIR:$AUTOMODEL_DIR/examples/sihanzeng:$SCRIPTS_DIR:/opt/megatron-lm/megatron/core/distributed/fsdp/src:$PYTHONPATH
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR

echo "=== Gemma4 26B-A4B-it SFT — Streaming Blended (Tulu3 0.8 + Step3.5 0.2) ==="
srun --ntasks=1 --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED,NCCL_DEBUG,CUDA_DEVICE_MAX_CONNECTIONS,PYTHONPATH,TORCH_NCCL_TRACE_BUFFER_SIZE \
     --container-mount-home \
     bash -c "
pip install torchdata -q 2>&1 | tail -1
cd $AUTOMODEL_DIR
torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/train_ft.py -c $CONFIG
"
