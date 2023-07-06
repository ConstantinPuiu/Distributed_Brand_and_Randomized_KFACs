#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=2G_RK_C
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
OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=2 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_SGD_torchrun_MCI.py --world_size 2 --n_epoch 10 --batch_size 256 \
--momentum 0.0 --WD 0.0007 \
--lr_schedule_type 'exp' --base_lr 0.3 --lr_decay_rate 9 --lr_decay_period 80 --auto_scale_forGPUs_and_BS 1 \
--test_at_end 1 --test_every_X_epochs 3 \
--seed 12345 --print_tqdm_progress_bar 1 \
--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
--net_type 'resnet18' \
--data_root_path '/data/math-opt-ml/' \
--dataset 'cifar10' \
--use_nesterov 0 \
--momentum_dampening_schedule_flag 0 --momentum_dampening 0 \
