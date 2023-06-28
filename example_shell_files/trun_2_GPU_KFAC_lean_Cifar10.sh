#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=2G_K_C
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=15G
#SBATCH --partition=devel

module purge
module load Anaconda3/2020.11
module load foss/2020a

nvidia-smi

source activate /data/math-opt-ml/chri5570/myenv
#optDLvenv

#mpiexec python ./attempt_2_GPUs_naive_KFAC.py
#NCCL_BLOCKING_WAIT=1
OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=2 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 2 --n_epoch 6 --batch_size 256 \
--test_at_end 1 --test_every_X_epochs 3 \
--net_type 'resnet18' \
--data_root_path '/data/math-opt-ml/' \
--dataset 'cifar10' \
--work_alloc_propto_EVD_cost 1 \
--TInv_schedule_flag 0 --TCov_schedule_flag 0 --KFAC_damping_schedule_flag 0

