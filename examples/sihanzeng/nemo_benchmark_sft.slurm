#!/bin/bash
#SBATCH --job-name=nemo_bench
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/nemo_bench_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel
SCRIPTS_DIR=$AUTOMODEL_DIR/examples/sihanzeng
CONFIG=$SCRIPTS_DIR/gemma4_26b_sft_benchmark.yaml

export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer_cache
export HF_HOME=/tmp/hf_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$AUTOMODEL_DIR:$SCRIPTS_DIR:/opt/megatron-lm/megatron/core/distributed/fsdp/src:$PYTHONPATH
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

mkdir -p $FLASHINFER_WORKSPACE_DIR $HF_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=13742
export NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))

echo "=== Gemma4 SFT Benchmark: $SLURM_NNODES nodes, $WORLD_SIZE GPUs, seq_len=8k ==="
echo "Target: match amaia 4B baseline of 1,812 tok/s/gpu"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-env=FLASHINFER_WORKSPACE_DIR,HF_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,PYTHONUNBUFFERED,NCCL_DEBUG,CUDA_DEVICE_MAX_CONNECTIONS,PYTHONPATH,TORCH_NCCL_TRACE_BUFFER_SIZE,MASTER_ADDR,MASTER_PORT \
     --container-mount-home \
     bash -c "
pip install torchdata -q 2>&1 | tail -1
cd $AUTOMODEL_DIR
torchrun --nproc-per-node=8 \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    nemo_automodel/recipes/llm/train_ft.py -c $CONFIG
"
