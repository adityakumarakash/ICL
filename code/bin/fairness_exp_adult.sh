#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

################### Running for multiple random seeds #######################

exp="fairness_adult"

for seed in 685 648 535 536 72 540 490 863 315 448
do
  # ICL
  a="-10"
  b="0.1"
  python src/fairness_exp.py \
                --experiment_name "$exp" \
                --run_type 'icl' \
                --dataset_name 'Adult' \
                --num_epochs 201 \
                --lr 1e-3 \
                --adv_lr 0.1 \
                --alpha 1.0 \
                --alpha_max 1000.0 \
                --alpha_gamma 1.5 \
                --comp_type 'icl' \
                --icl_a "$a" \
                --icl_b "$b" \
                --adv_num_epochs 200 \
                --seed "$seed"

  # UAI
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'uai' \
#               --dataset_name 'Adult' \
#               --num_epochs 201 \
#               --lr 1e-3 \
#               --adv_lr 0.1 \
#               --alpha 10.0 \
#               --alpha_max 10.0 \
#               --alpha_gamma 1.5 \
#               --comp_type 'uai' \
#               --adv_num_epochs 200 \
#               --disc_lr 0.1 \
#               --mask_lambda 1e-2 \
#               --seed "$seed" \
#               --gpu_id "0"

  # MMD Lap, MMD_f
#  python src/fairness_exp.py \
#                --experiment_name "$exp" \
#                --run_type 'mmd_lap' \
#                --dataset_name 'Adult' \
#                --num_epochs 201 \
#                --lr 1e-3 \
#                --adv_lr 0.1 \
#                --alpha 1.0 \
#                --alpha_max 1000.0 \
#                --alpha_gamma 1.5 \
#                --comp_type 'mmd_lap' \
#                --mmd_lap_p "10" \
#                --adv_num_epochs 200 \
#                --seed "$seed"

  # Adv, CAI
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'adv' \
#               --dataset_name 'Adult' \
#               --num_epochs 201 \
#               --lr 1e-3 \
#               --adv_lr 0.1 \
#               --alpha 1.0 \
#               --alpha_max 1.0 \
#               --alpha_gamma 1.5 \
#               --comp_type 'adv_training' \
#               --adv_num_epochs 200 \
#               --disc_lr 0.1 \
#               --seed "$seed"

#  # OT Pairwise
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'ot_pairwise' \
#               --dataset_name 'Adult' \
#               --num_epochs 201 \
#               --lr 1e-3 \
#               --adv_lr 0.1 \
#               --alpha 1.0 \
#               --alpha_max 100.0 \
#               --alpha_gamma 1.5 \
#               --comp_type 'ot_pairwise' \
#               --adv_num_epochs 200 \
#               --seed "$seed"

#  # MMD Experiment, MMD_s
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'mmd' \
#               --dataset_name 'Adult' \
#               --num_epochs 201 \
#               --lr 1e-3 \
#               --adv_lr 0.1 \
#               --alpha 1.0 \
#               --alpha_max 100.0 \
#               --alpha_gamma 1.5 \
#               --comp_type 'mmd_loss' \
#               --adv_num_epochs 200 \
#               --seed "$seed"

#  # MI, KL Experiment
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'kl' \
#               --dataset_name 'Adult' \
#               --num_epochs 201 \
#               --lr 1e-3 \
#               --adv_lr 0.1 \
#               --alpha 1.0 \
#               --alpha_max 1.0 \
#               --alpha_gamma 1.5 \
#               --comp_type 'kl' \
#               --adv_num_epochs 200 \
#               --seed "$seed"

  # Baseline experiment, unregularized
#  python src/fairness_exp.py \
#              --experiment_name "$exp" \
#              --run_type 'none' \
#              --dataset_name 'Adult' \
#              --num_epochs 201 \
#              --lr 1e-3 \
#              --adv_lr 0.1 \
#              --alpha 1.0 \
#              --alpha_max 100.0 \
#              --alpha_gamma 1.5 \
#              --comp_type 'none' \
#              --adv_num_epochs 200 \
#              --seed "$seed"
done