#!/bin/bash
# Multi-node Gemma4 26B MoE SFT with NeMo Automodel
# Usage: NODES=2 CONFIG=gemma4_26b_sft_packed_2node.yaml sbatch nemo_multinode_sft.slurm
#    or: NODES=4 CONFIG=gemma4_26b_sft_packed_4node.yaml sbatch nemo_multinode_sft.slurm

#SBATCH --job-name=nemo_multi
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/nemo_multi_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel
SCRIPTS_DIR=$AUTOMODEL_DIR/examples/sihanzeng
CONFIG_FILE=${CONFIG:-gemma4_26b_sft_packed.yaml}
CONFIG=$SCRIPTS_DIR/$CONFIG_FILE

# Rendezvous for multi-node torchrun
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

echo "=== Gemma4 26B MoE SFT — ${SLURM_NNODES} nodes ($(( SLURM_NNODES * 8 )) GPUs) ==="
echo "Config: $CONFIG_FILE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# Copy model from Lustre to local NVMe for faster loading
LOCAL_MODEL=/tmp/gemma4_model
LUSTRE_MODEL=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it

echo "Copying model to local disk..."
START_COPY=$(date +%s)
srun --ntasks-per-node=1 bash -c "
  mkdir -p $LOCAL_MODEL
  rsync -a --info=progress2 $LUSTRE_MODEL/ $LOCAL_MODEL/
  echo \"\$(hostname): copy done\"
"
END_COPY=$(date +%s)
echo "Model copy took $(( END_COPY - START_COPY ))s"

# Update config to use local model path
LOCAL_CONFIG=/tmp/sft_config_local.yaml
sed "s|pretrained_model_name_or_path:.*|pretrained_model_name_or_path: $LOCAL_MODEL|" $CONFIG > $LOCAL_CONFIG

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/tmp:/tmp \
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
    nemo_automodel/recipes/llm/train_ft.py -c $LOCAL_CONFIG
"
