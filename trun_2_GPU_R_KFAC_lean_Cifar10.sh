#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=2G_RK_C
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 --constraint='gpu_sku:V100'
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

#mpiexec python ./attempt_2_GPUs_naive_KFAC.py
#NCCL_BLOCKING_WAIT=1
OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=2 ./n_GPUs_dist_R_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 2 --n_epoch 10

