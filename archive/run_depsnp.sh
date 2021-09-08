#!/bin/bash
#SBATCH -A wmu101
#SBATCH --job-name="train"
#SBATCH --output="comet-out/mult-0.00001.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --constraint=exclusive
#SBATCH --no-requeue
#SBATCH --gres=gpu:p100:1
#SBATCH --wait=0
#SBATCH -t 6:00:00

#IN_FOLDER=train_lstm_mods_mass_hcd_all
IN_FOLDER=pred-full-deepnovo_p_o
echo "Copying data."
cp  /oasis/projects/nsf/wmu101/mtari008/DeepSNAP/data/$IN_FOLDER.tar.gz /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz
echo "Copied!"
echo "Extracting files."
tar xzf /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz -C /scratch/$USER/$SLURM_JOB_ID/
echo "Extraction done!"
ls /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER/
python3 main.py -s comet
