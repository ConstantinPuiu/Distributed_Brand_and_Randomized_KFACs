#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=4G_B_pure
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
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

#mpiexec python ./attempt_4_GPUs_naive_KFAC.py
#NCCL_LL_THRESHOLD=0
#OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=4 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/CIFAR10/n_GPUs_dist_B_R_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 4 --n_epochs 10 --brand_period 500000 --brand_update_multiplier_to_TCov 5
OMP_NUM_THREADS=8 torchrun --standalone --nnodes 1 --nproc_per_node=4 /home/chri5570/Distributed_Brand_and_Randomized_KFACs/main_files/CIFAR10/n_GPUs_dist_B_pure_KFAC_torchrun_lean_KFACTORS_CIFAR_10.py --world_size 4 --n_epochs 20 \
--brand_update_multiplier_to_TCov 5 \
--work_alloc_propto_RSVD_and_B_cost 1 \
--B_truncate_before_inversion 1 \
--adaptable_rsvd_rank 1 --rsvd_rank_adaptation_TInv_multiplier 1 \
--adaptable_B_rank 1 --B_rank_adaptation_T_brand_updt_multiplier 1 \
--TInv_schedule_flag 1 --TCov_schedule_flag 1 --brand_update_multiplier_to_TCov_schedule_flag 1 --KFAC_damping_schedule_flag 1
