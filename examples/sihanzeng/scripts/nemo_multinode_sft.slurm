#!/bin/bash
# Multi-node Gemma4 26B MoE SFT with NeMo Automodel
# Usage: CONFIG=configs/gemma4_sft_tulu3.yaml sbatch --nodes=1 scripts/nemo_multinode_sft.slurm
#    or: CONFIG=configs/gemma4_sft_stream_packed.yaml sbatch --nodes=4 scripts/nemo_multinode_sft.slurm

#SBATCH --job-name=nemo_multi
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/nemo_multi_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel
SIHANZENG_DIR=$AUTOMODEL_DIR/examples/sihanzeng
CONFIG_FILE=${CONFIG:-configs/gemma4_sft_stream_packed.yaml}
CONFIG=$SIHANZENG_DIR/$CONFIG_FILE

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_NODEID}
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$AUTOMODEL_DIR:$SIHANZENG_DIR/dataloader:/opt/megatron-lm/megatron/core/distributed/fsdp/src:$PYTHONPATH
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_IB_TIMEOUT=120
export NCCL_SOCKET_TIMEOUT=120000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR

echo "=== Gemma4 26B MoE SFT — ${SLURM_NNODES} nodes ($(( SLURM_NNODES * 8 )) GPUs) ==="
echo "Config: $CONFIG_FILE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED,NCCL_DEBUG,CUDA_DEVICE_MAX_CONNECTIONS,PYTHONPATH,TORCH_NCCL_TRACE_BUFFER_SIZE,NCCL_IB_TIMEOUT,NCCL_SOCKET_TIMEOUT,TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
     --container-mount-home \
     bash -c "
pip install torchdata -q 2>&1 | tail -1
cd $AUTOMODEL_DIR
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    nemo_automodel/recipes/llm/train_ft.py -c $CONFIG
"
