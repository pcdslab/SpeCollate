#!/bin/bash
#SBATCH --job-name="atles-dragon"
#SBATCH --output="atles-out/bak/atles.%j.%N.out"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=92gb
#SBATCH --account=fsaeed
#SBATCH --no-requeue
#SBATCH --time=72:00:00

mkdir atles-out/$SLURM_JOB_ID
mkdir atles-out/$SLURM_JOB_ID/models
mkdir atles-out/$SLURM_JOB_ID/code
cp -R src config.ini read_spectra.py read_spectra.sh run_train.py train.sh atles-out/$SLURM_JOB_ID/code/

# CUDA_LAUNCH_BLOCKING=1 python3 run_train.py
python3 -u run_train.py
