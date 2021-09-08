#!/bin/bash

SLURM_JOB_ID=37154933
#CUDA_VISIBLE_DEVICES=1

IN_FOLDER=pred-full-deepnovo
echo "Copying data."
#cp  /oasis/projects/nsf/wmu101/mtari008/DeepSNAP/data/$IN_FOLDER.tar.gz /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz
echo "Copied!"
echo "Extracting files."
#tar xzf /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER.tar.gz -C /scratch/$USER/$SLURM_JOB_ID/
echo "Extraction done!"
ls /scratch/$USER/$SLURM_JOB_ID/$IN_FOLDER/
CUDA_VISIBLE_DEVICES=1 python3 main.py -s comet

