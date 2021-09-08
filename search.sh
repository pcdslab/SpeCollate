#!/bin/bash
#SBATCH --job-name="dbsearch"
#SBATCH --output="expanse-out/dbsearch.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=10
#SBATCH --mem=192G
#SBATCH --account=wmu101
#SBATCH --no-requeue
#SBATCH -t 04:00:00

module purge
module load gpu
module load slurm

#IN_FOLDER=train_lstm_mods_mass_hcd_all
MGF_DIR=/expanse/lustre/projects/wmu101/mtari008/DeepSNAP/data/pxd000612/human-hcd-phospho-mgf

PREP_PATH=/expanse/lustre/projects/wmu101/mtari008/DeepSNAP/data/preprocessed
PREP_FILE=humanc-hcd-phospho-no-ch-mass.tar.gz

PEP_DIR=/expanse/lustre/projects/wmu101/mtari008/DeepSNAP/data/peps

PERCOLATOR_DIR=/expanse/lustre/projects/wmu101/mtari008/DeepSNAP/percolator

SCRATCH_LOC=/scratch/$USER/job_$SLURM_JOB_ID

(cd $SCRATCH_LOC && mkdir percolator) # Create directory for percolator files.

echo "Copying MGFs..."
cp -R $MGF_DIR $SCRATCH_LOC/

echo "Copying Preprocessed Spectra..."
cp $PREP_PATH/$PREP_FILE $SCRATCH_LOC/

echo "Extracting Preprocessed Spectra..."
tar xzf $SCRATCH_LOC/$PREP_FILE -C $SCRATCH_LOC

echo "Copying Peptides..."
cp -R $PEP_DIR $SCRATCH_LOC/

ls $SCRATCH_LOC

echo "Executing run_search.py..."
# CUDA_LAUNCH_BLOCKING=1 python3 run_train.py
python3 run_search.py -p False

(cd $SCRATCH_LOC/percolator && crux percolator target.pin decoy.pin --list-of-files T --overwrite T)

cp $SCRATCH_LOC/percolator/target.pin $PERCOLATOR_DIR/target.$SLURM_JOB_ID.pin
cp $SCRATCH_LOC/percolator/decoy.pin $PERCOLATOR_DIR/decoy.$SLURM_JOB_ID.pin

cp $SCRATCH_LOC/percolator/crux-output/percolator.target.peptides.txt $PERCOLATOR_DIR/peps-$SLURM_JOB_ID.txt