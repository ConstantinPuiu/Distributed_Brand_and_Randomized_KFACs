#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --job-name=t_2GPU_K_trial
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

#mpiexec python ./attempt_2_GPUs_naive_KFAC.py
torchrun --standalone --nnodes 1 --nproc_per_node=2 ./n_GPUs_dist_KFAC_torchrun.py --world_size 2

