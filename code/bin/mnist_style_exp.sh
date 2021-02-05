#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

exp_name='mnist_style'

for seed in 685 648 535 536 72 540 490 863 315 448
do
  # ICL
  python src/mnist_style_exp.py \
                 --experiment_name "$exp_name" \
                 --debug_tag 'icl' \
                 --comp_type 'icl' \
                 --icl_a 5 \
                 --icl_b 0.5 \
                 --comp_lambda 0.01 \
                 --comp_lambda_max 1.0 \
                 --comp_gamma 1.5 \
                 --num_epochs 100 \
                 --adv_num_epochs 200 \
                 --adv_lr 0.0005 \
                 --seed "$seed"

  # CAI
#  python src/mnist_style_exp.py \
#                    --experiment_name "$exp_name" \
#                    --debug_tag 'adv' \
#                    --comp_type 'adv_training' \
#                    --comp_lambda 1.0 \
#                    --comp_lambda_max 1.0 \
#                    --comp_gamma 1.5 \
#                    --num_epochs 100 \
#                    --adv_num_epochs 200 \
#                    --adv_lr 0.0005 \
#                    --disc_lr 0.001 \
#                    --seed "$seed"

  # Pairwise OT
#  python src/mnist_style_exp.py \
#                    --experiment_name "$exp_name" \
#                    --debug_tag 'ot_pairwise' \
#                    --comp_type 'ot_pairwise' \
#                    --comp_lambda 0.1 \
#                    --comp_lambda_max 500.0 \
#                    --comp_gamma 1.5 \
#                    --num_epochs 100 \
#                    --adv_num_epochs 200 \
#                    --adv_lr 0.0005 \
#                    --seed "$seed"

  # MMD_f, lap_mmd
#  python src/mnist_style_exp.py \
#                   --experiment_name "$exp_name" \
#                   --debug_tag 'lap_mmd' \
#                   --comp_type 'mmd_lap' \
#                   --comp_lambda 0.01 \
#                   --comp_lambda_max 100.0 \
#                   --comp_gamma 1.5 \
#                   --mmd_lap_p 2.0 \
#                   --num_epochs 100 \
#                   --adv_num_epochs 200 \
#                   --adv_lr 0.0005 \
#                   --seed "$seed"

  # MMD_s
#  python src/mnist_style_exp.py \
#                  --experiment_name "$exp_name" \
#                  --debug_tag 'mmd' \
#                  --comp_type 'mmd_loss' \
#                  --comp_lambda 0.01 \
#                  --comp_lambda_max 1000.0 \
#                  --comp_gamma 1.5 \
#                  --num_epochs 100 \
#                  --adv_num_epochs 200 \
#                  --adv_lr 0.0005 \
#                  --seed "$seed"

  # KL, MI baseline
#  python src/mnist_style_exp.py \
#                --experiment_name "$exp_name" \
#                --debug_tag 'kl' \
#                --comp_type 'kl' \
#                --comp_lambda 0.01 \
#                --comp_lambda_max 0.1 \
#                --comp_gamma 1.5 \
#                --num_epochs 100 \
#                --adv_num_epochs 200 \
#                --adv_lr 0.0005 \
#                --seed "$seed"

  # Baseline unregularized
#  python src/mnist_style_exp.py \
#                --experiment_name "$exp_name" \
#                --debug_tag 'none' \
#                --comp_type 'none' \
#                --num_epochs 100 \
#                --adv_num_epochs 200 \
#                --adv_lr 0.0005 \
#                --seed "$seed"
done