#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/media/compute/homes/dmindlin/xray/slurm_logs/transform_dataset.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun transform_dataset.job.sh