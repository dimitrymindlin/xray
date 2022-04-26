#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/dmindlin/xray/slurm_logs/transform_dataset.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun transform_dataset.job.sh