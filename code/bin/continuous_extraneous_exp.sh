#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

exp='continuous_extraneous'

for seed in 685 648 535 536 72 540 490 863 315 448
do
  # ICL
  python src/continuous_extraneous_exp.py \
                   --experiment_name "$exp" \
                   --run_type 'icl' \
                   --dataset_name 'Adult' \
                   --num_epochs 201 \
                   --lr 1e-3 \
                   --adv_lr 0.1 \
                   --alpha 0.1 \
                   --alpha_max "1000" \
                   --alpha_gamma 1.1 \
                   --comp_type 'icl' \
                   --icl_a "-5.0" \
                   --icl_b "10.0" \
                   --adv_num_epochs 200 \
                   --neighbour_threshold "0.05" \
                   --seed "$seed"

  # UAI
#  python src/continuous_extraneous_exp.py \
#                  --experiment_name "$exp" \
#                  --run_type 'uai' \
#                  --dataset_name 'Adult' \
#                  --num_epochs 201 \
#                  --lr 1e-3 \
#                  --adv_lr 0.1 \
#                  --alpha 1.0 \
#                  --alpha_max 1.0 \
#                  --alpha_gamma 1.5 \
#                  --comp_type 'uai' \
#                  --adv_num_epochs 200 \
#                  --disc_lr 0.1 \
#                  --seed "$seed" \
#                  --gpu_ids "0"

  # CAI, Adv
#  python src/continuous_extraneous_exp.py \
#                  --experiment_name "$exp" \
#                  --run_type 'adv' \
#                  --dataset_name 'Adult' \
#                  --num_epochs 201 \
#                  --lr 1e-3 \
#                  --adv_lr 0.1 \
#                  --alpha 0.1 \
#                  --alpha_max 1.0 \
#                  --alpha_gamma 1.5 \
#                  --comp_type 'adv_training' \
#                  --adv_num_epochs 200 \
#                  --disc_lr 0.1 \
#                  --disc_hidden_layers 1 \
#                  --seed "$seed"

  # Baseline
#  python src/continuous_extraneous_exp.py \
#                 --experiment_name "$exp" \
#                 --run_type 'none' \
#                 --dataset_name 'Adult' \
#                 --num_epochs 201 \
#                 --lr 1e-3 \
#                 --adv_lr 0.1 \
#                 --alpha 1.0 \
#                 --alpha_max 1.0 \
#                 --alpha_gamma 1.5 \
#                 --comp_type 'none' \
#                 --adv_num_epochs 200 \
#                 --seed "$seed"
done