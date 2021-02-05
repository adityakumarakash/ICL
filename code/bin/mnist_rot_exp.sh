#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

exp_name='mnist_rotation'
gpu="0"

for seed in 685 648 535 536 72 540 490 863 315 448
do
  #  ICL
  a=1.0
  b=0.5
  lambda=50.0
  python src/mnist_rot_exp.py \
                --experiment_name "$exp_name" \
                --dataset_name "RandomRotMNIST" \
                --model_name "MNIST_EncDec" \
                --run_type 'icl' \
                --num_epochs 120 \
                --lr 1e-4 \
                --adv_lr 1e-3 \
                --alpha "$lambda" \
                --alpha_max "$lambda" \
                --alpha_gamma 1.5 \
                --comp_type 'icl' \
                --adv_num_epochs 100 \
                --latent_dim 10 \
                --icl_a "$a" \
                --icl_b "$b" \
                --latent_dim2 20 \
                --beta 0.1 \
                --pred_lambda 100.0 \
                --batch_size 64 \
                --adv_batch_size 64 \
                --gpu_ids "$gpu" \
                --seed "$seed"


  # Unregularized Case
#  python src/mnist_rot_exp.py \
#               --experiment_name "$exp_name" \
#               --dataset_name "RandomRotMNIST" \
#               --model_name "MNIST_EncDec" \
#               --run_type 'none' \
#               --num_epochs 120 \
#               --lr 1e-4 \
#               --adv_lr 1e-3 \
#               --comp_type 'none' \
#               --adv_num_epochs 100 \
#               --latent_dim 10 \
#               --latent_dim2 20 \
#               --beta 0.1 \
#               --pred_lambda 100.0 \
#               --batch_size 64 \
#               --adv_batch_size 64 \
#               --save_step 100 \
#               --seed "$seed" \
#               --gpu_ids "1"


   #  MMD_f, MMD Laplacian
#  a=1.0
#  b=0.5
#  lambda=1.0
#  python src/rot_mnist_det.py \
#                --experiment_name "$exp_name" \
#                --dataset_name "RandomRotMNIST" \
#                --model_name "MNIST_EncDec" \
#                --run_type 'lap_mmd' \
#                --num_epochs 120 \
#                --lr 1e-4 \
#                --adv_lr 1e-3 \
#                --alpha "$lambda" \
#                --alpha_max "$lambda" \
#                --alpha_gamma 1.5 \
#                --comp_type 'mmd_lap' \
#                --adv_num_epochs 100 \
#                --latent_dim 10 \
#                --icl_a "$a" \
#                --icl_b "$b" \
#                --latent_dim2 20 \
#                --beta 0.1 \
#                --pred_lambda 100.0 \
#                --batch_size 64 \
#                --adv_batch_size 64 \
#                --gpu_ids "$gpu" \
#                --seed "$seed"
done