#!/bin/bash
#SBATCH --job-name=debug_dl
#SBATCH --account=ads_cpt
#SBATCH --partition=a4
#SBATCH --qos=ads_cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=0
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/logs/debug_dl_%j.log

CONTAINER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo-20260401.sqsh
AUTOMODEL_DIR=$HOME/Automodel

srun --container-image=$CONTAINER \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre \
     --container-mount-home \
     bash -c "
cd $AUTOMODEL_DIR
python3 examples/sihanzeng/debug_dataloader.py
"
