#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=all_1G_CH
#SBATCH --nodes=1 --constraint=fabric:HDR
#SBATCH --gres=gpu:1 --constraint='gpu_sku:V100-SXM2' --constraint='gpu_mem:32GB'
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

#####################################################################
############## Run SGD 5 times # 35 mins x 5 required ###############
#####################################################################
# NOTE: All other solvers need at max 30 epochs for 92.% acc (actually 18-20 is enough, but SGD needs more!, running 70 epochs for SGD, 30 for all other solvers)
# NOTE: For SGD best lr schdeule is with exp rather than with staricase: using an exponential decay with rapid decay factor, slow decay period, and just the first part
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_SGD_torchrun_MCI.py --world_size 1 --n_epoch 105 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--momentum 0.9 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.1 --lr_decay_rate 3 --lr_decay_period 50 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--use_nesterov 0 \
	--momentum_dampening_schedule_flag 0 --momentum_dampening 0 

	sleep 1m 1s
done
#####################################################################
############## Run KFAC 5 times # 40 mins x 5 required ##############
#####################################################################
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 1 --n_epoch 30 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.3 --lr_decay_rate 3 --lr_decay_period 12 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--TInv_period 100 --TCov_period 20 \
	--work_alloc_propto_EVD_cost 1 \
	--TInv_schedule_flag 0 --TCov_schedule_flag 0 --KFAC_damping_schedule_flag 0
	sleep 1m 1s
done
### pause 5 mins for cooldown ?
sleep 1m 1s

#####################################################################
############## Run R-KFAC 5 times # 10mins x 5 required #############
#####################################################################
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_R_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 1 --n_epoch 30 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.3 --lr_decay_rate 3 --lr_decay_period 12 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--TInv_period 100 --TCov_period 20 \
	--work_alloc_propto_RSVD_cost 1 --work_eff_alloc_with_time_measurement 0 \
	--adaptable_rsvd_rank 0 --rsvd_rank_adaptation_TInv_multiplier 1 \
	--TInv_schedule_flag 0 --TCov_schedule_flag 0 --KFAC_damping_schedule_flag 0
	
	sleep 1m 1s
done
### pause 5 mins for cooldown 
sleep 1m 1s

#####################################################################
############## Run B-KFAC 5 times # 10mins x 5 required #############
#####################################################################
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_B_pure_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 1 --n_epochs 30 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.3 --lr_decay_rate 3 --lr_decay_period 12 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--TInv_period 100 --TCov_period 20 \
	--brand_update_multiplier_to_TCov 5 \
	--work_alloc_propto_RSVD_and_B_cost 1 \
	--B_truncate_before_inversion 1 --adaptable_rsvd_rank 1 \
	--rsvd_rank_adaptation_TInv_multiplier 1 --adaptable_B_rank 1 \
	--B_rank_adaptation_T_brand_updt_multiplier 1 \
	--TInv_schedule_flag 0 --TCov_schedule_flag 0 --brand_update_multiplier_to_TCov_schedule_flag 0 --KFAC_damping_schedule_flag 0
	
	sleep 1m 1s
done
### pause 5 mins for cooldown 
sleep 1m 1s

#####################################################################
############## Run BR-KFAC 5 times # 10mins x 5 required ############
#####################################################################
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_B_R_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 1 --n_epochs 30 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.3 --lr_decay_rate 3 --lr_decay_period 12 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--TInv_period 100 --TCov_period 20 \
	--brand_update_multiplier_to_TCov 1 \
	--B_R_period 5 \
	--B_truncate_before_inversion 1 \
	--work_alloc_propto_RSVD_and_B_cost 1 \
	--adaptable_rsvd_rank 1 --rsvd_rank_adaptation_TInv_multiplier 1 \
	--adaptable_B_rank 1 --B_rank_adaptation_T_brand_updt_multiplier 1 \
	--TInv_schedule_flag 0 --TCov_schedule_flag 0 --brand_update_multiplier_to_TCov_schedule_flag 0 --B_R_period_schedule_flag 0 --KFAC_damping_schedule_flag 0
	
	sleep 1m 1s
done
### pause 5 mins for cooldown 
sleep 1m 1s

########################################################################
############## Run BRC-KFAC 5 times # 10mins x 5 required ##############
########################################################################
for SEED in 12345 23456 34567 45678 56789
do
	OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=1 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/n_GPUs_dist_B_R_C_KFAC_torchrun_lean_KFACTORS_MCI.py --world_size 1 --n_epochs 30 --batch_size 128 \
	--stop_at_test_acc 1 --stopping_test_acc 70.00 \
	--kfac_clip 0.07 --stat_decay 0.95 --momentum 0.0 --WD 0.0007 \
	--lr_schedule_type 'staircase' --base_lr 0.3 --lr_decay_rate 3 --lr_decay_period 12 --auto_scale_forGPUs_and_BS 1 \
	--test_at_end 1 --test_every_X_epochs 1 \
	--seed $SEED --print_tqdm_progress_bar 1 \
	--store_and_save_metrics 1 --metrics_save_path '/data/math-opt-ml/saved_metrics/' \
	--net_type 'VGG16_bn_lmxp' \
	--data_root_path '/data/math-opt-ml/' \
	--dataset 'cifar100' \
	--TInv_period 100 --TCov_period 20 \
	--brand_update_multiplier_to_TCov 1 \
	--B_R_period 5 \
	--B_truncate_before_inversion 1 \
	--work_alloc_propto_RSVD_and_B_cost 1 \
	--adaptable_rsvd_rank 1 --rsvd_rank_adaptation_TInv_multiplier 1 \
	--adaptable_B_rank 1 --B_rank_adaptation_T_brand_updt_multiplier 1 \
	--correction_multiplier_TCov 5 --brand_corection_dim_frac 0.2 \
	--TInv_schedule_flag 0 --TCov_schedule_flag 0 --brand_update_multiplier_to_TCov_schedule_flag 0 --B_R_period_schedule_flag 0 --correction_multiplier_TCov_schedule_flag 0 --KFAC_damping_schedule_flag 0
	
	sleep 1m 1s
done

