#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=8G_BRK_C
#SBATCH --nodes=2
#SBATCH --gres=gpu:4 --constraint='gpu_sku:A100'
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
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
#NCCL_BLOCKING_WAIT=1
OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=4 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_B_R_KFAC_torchrun_lean_KFACTORS_MCI.py \
--world_size 8 --n_epochs 20 --batch_size 256 \
--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
--lr_schedule_type 'exp' --base_lr 0.3 --lr_decay_rate 9 --lr_decay_period 80 --auto_scale_forGPUs_and_BS 1 \
--test_at_end 1 --test_every_X_epochs 3 \
--seed 12345 --print_tqdm_progress_bar 1 \
--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
--net_type 'resnet18' \
--data_root_path '/data/math-opt-ml/' \
--dataset 'cifar10' \
--TInv_period 100 --TCov_period 20 \
--brand_update_multiplier_to_TCov 1 \
--B_R_period 5 \
--B_truncate_before_inversion 1 \
--work_alloc_propto_RSVD_and_B_cost 1 \
--adaptable_rsvd_rank 1 --rsvd_rank_adaptation_TInv_multiplier 1 \
--adaptable_B_rank 1 --B_rank_adaptation_T_brand_updt_multiplier 1 \
--TInv_schedule_flag 0 --TCov_schedule_flag 0 --brand_update_multiplier_to_TCov_schedule_flag 0 --B_R_period_schedule_flag 0 --KFAC_damping_schedule_flag 0
