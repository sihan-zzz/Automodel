#!/bin/bash
# Same as nemo_multinode_sft.slurm but WITHOUT local model copy.
# Tests whether loading directly from Lustre is faster than copy-then-load.
# Usage: CONFIG=fair_stream.yaml sbatch --nodes=1 nemo_sft_nocopy.slurm

#SBATCH --job-name=nemo_nocopy
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/nemo_nocopy_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel
SCRIPTS_DIR=$AUTOMODEL_DIR/examples/sihanzeng
CONFIG_FILE=${CONFIG:-fair_stream.yaml}
CONFIG=$SCRIPTS_DIR/$CONFIG_FILE

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_NODEID}
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$AUTOMODEL_DIR:$SCRIPTS_DIR:/opt/megatron-lm/megatron/core/distributed/fsdp/src:$PYTHONPATH
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_IB_TIMEOUT=120
export NCCL_SOCKET_TIMEOUT=120000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR

echo "=== Gemma4 26B MoE SFT (NO LOCAL COPY) — ${SLURM_NNODES} nodes ($(( SLURM_NNODES * 8 )) GPUs) ==="
echo "Config: $CONFIG_FILE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "Loading model directly from Lustre (no local copy)"

START_TRAIN=$(date +%s)

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

END_TRAIN=$(date +%s)
echo "Total wall time: $(( END_TRAIN - START_TRAIN ))s"
