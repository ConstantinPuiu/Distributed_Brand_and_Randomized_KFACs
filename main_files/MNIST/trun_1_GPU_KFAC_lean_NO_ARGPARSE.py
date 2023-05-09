#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=t_1G_KNoA
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 --constraint='gpu_sku:RTX'
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=15G
#SBATCH --partition=short

module purge
module load Anaconda3/2020.11
module load foss/2020a

nvidia-smi

source activate /data/math-opt-ml/chri5570/myenv
#optDLvenv

#mpiexec python ./attempt_4_GPUs_naive_KFAC.py
torchrun --standalone --nnodes 1 --nproc_per_node=1 ./1_GPUs_dist_KFAC_torchrun_lean_KFACTORS_NO_ARGPARSE.py

