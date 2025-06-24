#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=45:00:00
#SBATCH --mem=256GB
#SBATCH --mail-type=END
#SBATCH --mail-user=yc3400@nyu.edu
#SBATCH --output=/home/yc3400/scratch_research/gitrepo/AffineInvariantSamplers/logs/%A_%a.out
#SBATCH --error=/home/yc3400/scratch_research/gitrepo/AffineInvariantSamplers/logs/%A_%a.err

## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/yc3400/my_env/overlay-15GB-500K.ext3:ro
LAUNCHER_FILE=/home/yc3400/scratch_research/gitrepo/AffineInvariantSamplers/experiments_Gaussian-slurm.py
PY_DIR=/home/yc3400/scratch_research/gitrepo/AffineInvariantSamplers
output_folder=/home/yc3400/scratch_research/gitrepo/AffineInvariantSamplers/logs

## modules provided by the singularity image
module purge

singularity exec --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; \
        cd $PY_DIR; \
        python $LAUNCHER_FILE"
exit
