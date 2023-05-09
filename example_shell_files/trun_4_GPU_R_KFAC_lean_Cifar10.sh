#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=4G_RK_C
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=15G
#SBATCH --partition=medium

module purge
module load Anaconda3/2020.11
module load foss/2020a

nvidia-smi

source activate /data/math-opt-ml/chri5570/myenv
#optDLvenv

#mpiexec python ./attempt_4_GPUs_naive_KFAC.py
OMP_NUM_THREADS=8 NCCL_LL_THRESHOLD=0 torchrun --standalone --nnodes 1 --nproc_per_node=4 ./n_GPUs_dist_R_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 4 --n_epoch 10
