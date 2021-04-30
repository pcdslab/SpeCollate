#!/bin/bash
#SBATCH --job-name="wandb-test"
#SBATCH --output="expanse-out/specollate.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=96G
#SBATCH --account=wmu101
#SBATCH --no-requeue
#SBATCH -t 24:00:00

module purge
module load gpu
module load slurm

#IN_FOLDER=train_lstm_mods_mass_hcd_all
IN_FOLDER=train_lstm_mods_mass_hcd_all
echo "Copying data."
cp  /expanse/lustre/projects/wmu101/mtari008/DeepSNAP/data/train_ready/$IN_FOLDER.tar.gz /scratch/$USER/job_$SLURM_JOB_ID/$IN_FOLDER.tar.gz
echo "Copied!"
echo "Extracting files."
tar xzf /scratch/$USER/job_$SLURM_JOB_ID/$IN_FOLDER.tar.gz -C /scratch/$USER/job_$SLURM_JOB_ID/
echo "Extraction done!"
ls /scratch/$USER/job_$SLURM_JOB_ID/$IN_FOLDER/
#python3 -m torch.utils.bottleneck main.py
# CUDA_LAUNCH_BLOCKING=1 python3 run_train.py
python3 main.py