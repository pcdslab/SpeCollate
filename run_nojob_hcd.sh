#!/bin/bash

SLURM_JOB_ID=34553626
#CUDA_VISIBLE_DEVICES=1

IN_FOLDER=no_deepnovo_mods_train_lstm_hcd
time (echo "Copying data."
#cp  /oasis/projects/nsf/wmu101/mtari008/DeepSNAP/data/$IN_FOLDER.tar.gz /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz
echo "Copied!"
echo "Extracting files."
#tar xzf /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz -C /scratch/$USER/$SLURM_JOB_ID/
echo "Extraction done!"
ls /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER/
module load singularity
CUDA_VISIBLE_DEVICES=3 singularity exec --nv --bind /oasis/projects/nsf/wmu101/mtari008/DeepSNAP/:/DeepSNAP --bind /scratch/$USER/$SLURM_JOB_ID/:/scratch /oasis/projects/nsf/wmu101/mtari008/containers/cuda-sing.sif python3 /DeepSNAP/main.py -s comet) 
