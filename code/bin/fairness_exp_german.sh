#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

exp='german_fairness'

for seed in 685 648 535 536 72 540 490 863 315 448
do
  # ICL
  python src/fairness_exp.py \
                 --experiment_name "$exp" \
                 --run_type 'icl' \
                 --dataset_name 'German' \
                 --num_epochs 501 \
                 --lr 1e-2 \
                 --adv_lr 1e-2 \
                 --alpha 1.0 \
                 --alpha_max 50.0 \
                 --alpha_gamma 1.1 \
                 --comp_type 'icl' \
                 --icl_a 5.0 \
                 --icl_b 0.1 \
                 --adv_num_epochs 200 \
                 --seed "$seed"

  # UAI
#  python src/fairness_exp.py \
#                   --experiment_name "$exp" \
#                   --run_type 'uai' \
#                   --dataset_name 'German' \
#                   --num_epochs 501 \
#                   --lr 1e-2 \
#                   --adv_lr 1e-2 \
#                   --alpha 0.01 \
#                   --alpha_max 0.01 \
#                   --alpha_gamma 1.1 \
#                   --comp_type 'uai' \
#                   --adv_num_epochs 200 \
#                   --disc_lr 1e-2 \
#                   --mask_lambda 1e-2 \
#                   --seed "$seed" \
#                   --gpu_ids "0"

  # CAI, Adv
#  python src/fairness_exp.py \
#                   --experiment_name "$exp" \
#                   --run_type 'adv' \
#                   --dataset_name 'German' \
#                   --num_epochs 501 \
#                   --lr 1e-2 \
#                   --adv_lr 1e-2 \
#                   --alpha 0.1 \
#                   --alpha_max 0.1 \
#                   --alpha_gamma 1.1 \
#                   --comp_type 'adv_training' \
#                   --adv_num_epochs 200 \
#                   --disc_lr 1e-2 \
#                   --seed "$seed"

  # MMD Lap, MMD_f
#  python src/fairness_exp.py \
#                  --experiment_name "$exp" \
#                  --run_type 'lap_mmd' \
#                  --dataset_name 'German' \
#                  --num_epochs 501 \
#                  --lr 1e-2 \
#                  --adv_lr 1e-2 \
#                  --alpha 0.1 \
#                  --alpha_max 25.0 \
#                  --alpha_gamma 1.1 \
#                  --comp_type 'mmd_lap' \
#                  --mmd_lap_p 10.0 \
#                  --adv_num_epochs 200 \
#                  --seed "$seed"

  # MMD
#  python src/fairness_exp.py \
#                --experiment_name "$exp" \
#                --run_type 'mmd' \
#                --dataset_name 'German' \
#                --num_epochs 501 \
#                --lr 1e-2 \
#                --adv_lr 1e-2 \
#                --alpha 0.1 \
#                --alpha_max 2.0 \
#                --alpha_gamma 1.1 \
#                --comp_type 'mmd_loss' \
#                --adv_num_epochs 200 \
#                --seed "$seed"

  # OT Pairwise
#  python src/fairness_exp.py \
#               --experiment_name "$exp" \
#               --run_type 'ot_pairwise' \
#               --dataset_name 'German'\
#                --num_epochs 501 \
#                --lr 1e-2 \
#                --adv_lr 1e-2 \
#                --alpha 0.1 \
#                --alpha_max 100.0 \
#                --alpha_gamma 1.1 \
#                --comp_type 'ot_pairwise' \
#                --adv_num_epochs 200 \
#                --seed "$seed"

  # KL
#  python src/fairness_exp.py \
#              --experiment_name "$exp" \
#              --run_type 'kl' \
#              --dataset_name 'German' \
#              --num_epochs 501 \
#              --lr 1e-2 \
#              --adv_lr 1e-2 \
#              --alpha 0.1 \
#              --alpha_max 50.0 \
#              --alpha_gamma 1.1 \
#              --comp_type 'kl' \
#              --adv_num_epochs 200 \
#              --seed "$seed"


  # Baseline
#  python src/fairness_exp.py \
#              --experiment_name "$exp" \
#              --run_type 'none' \
#              --dataset_name 'German' \
#              --num_epochs 501 \
#              --lr 1e-2 \
#              --adv_lr 1e-2 \
#              --alpha 1.0 \
#              --alpha_max 1.0 \
#              --alpha_gamma 1.1 \
#              --comp_type 'none' \
#              --adv_num_epochs 200 \
#              --seed "$seed"
done