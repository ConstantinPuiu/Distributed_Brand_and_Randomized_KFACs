#!/bin/bash

#SBATCH --time=00:07:00
#SBATCH --job-name=2G_B_pr
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
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
OMP_NUM_THREADS=8 NCCL_BLOCKING_WAIT=1 torchrun --standalone --nnodes 1 --nproc_per_node=2 ./n_GPUs_dist_B_R_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 2 --n_epochs 10 --brand_period 5000000000000000 --brand_update_multiplier_to_TCov 5
