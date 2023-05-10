#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --job-name=1G_RK
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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
torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/CIFAR10/n_GPUs_dist_R_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 1 --n_epoch 10

