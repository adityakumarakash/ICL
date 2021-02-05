#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC="src/adni_classification"


# ICL regularized training
experiment_name="adni_icl_regularizer"
for split in 0 1 2 3 4
do
  logdir="result/${experiment_name}"
  python "${SRC}"/main.py \
          --logdir "$logdir" \
          --fold "$split" \
          --lr 3e-4 \
          --blocks 2 2 2 2 \
          --channels 16 16 32 64 \
          --use_mmd \
          --mmd_type 'icl' \
          --alpha 0.0001 \
          --alpha_max 0.1 \
          --alpha_gamma 1.2 \
          --max_epochs 150

  python "${SRC}"/run_adv_training.py \
          --experiment_name "${experiment_name}_adv" \
          --fold "${split}" \
          --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
          --adv_num_epochs 200
done


## Unregularized training
#experiment_name="adni_regularizer_none"
#for split in 0 1 2 3 4
#do
#  logdir="result/${experiment_name}"
#  python "${SRC}"/main.py \
#          --logdir "$logdir" \
#          --fold "$split" \
#          --lr 3e-4 \
#          --blocks 2 2 2 2 \
#          --channels 16 16 32 64 \
#          --max_epochs 150
#
#  python "${SRC}"/run_adv_training.py \
#          --experiment_name "${experiment_name}_adv" \
#          --fold "${split}" \
#          --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#          --adv_num_epochs 200
#done


# CAI, Adv training
#experiment_name="adni_regularize_cai"
#for split in 0 1 2 3 4
#do
#  logdir="result/${experiment_name}"
#  python "${SRC}"/main.py \
#          --logdir "$logdir" \
#          --fold "$split" \
#          --lr 3e-4 \
#          --blocks 2 2 2 2 \
#          --channels 16 16 32 64 \
#          --max_epochs 150 \
#          --use_adv
#
#  python "${SRC}"/run_adv_training.py \
#          --experiment_name "${experiment_name}_adv" \
#          --fold "${split}" \
#          --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#          --adv_num_epochs 200
#done


#experiment_name="adni_regularizer_mmd"
#for split in 0 1 2 3 4
#do
#    logdir="result/${experiment_name}"
#    python "${SRC}"/main.py \
#          --logdir "$logdir" \
#          --fold "$split" \
#          --lr 3e-4 \
#          --blocks 2 2 2 2 \
#          --channels 16 16 32 64 \
#          --max_epochs 150 \
#          --use_mmd \
#          --mmd_type 'mmd' \
#          --alpha 0.0001 \
#          --alpha_max 0.1 \
#          --alpha_gamma 1.2
#
#    python "${SRC}"/run_adv_training.py \
#          --experiment_name "${experiment_name}_adv" \
#          --fold "${split}" \
#          --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#          --adv_num_epochs 200
#done



#experiment_name="adni_regularize_mmd_lap"
#for split in 0
#do
#    logdir="result/${experiment_name}"
#    python "${SRC}"/main.py \
#           --logdir "$logdir" \
#           --fold "$split" \
#           --lr 3e-4 \
#           --blocks 2 2 2 2 \
#           --channels 16 16 32 64 \
#           --max_epochs 150 \
#           --use_mmd \
#           --mmd_type 'mmd_lap' \
#           --alpha 0.0001 \
#           --alpha_max 0.1 \
#           --alpha_gamma 1.2
#
#    python "${SRC}"/run_adv_training.py \
#           --experiment_name "${experiment_name}_adv" \
#           --fold "${split}" \
#           --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#           --adv_num_epochs 200
#done



#experiment_name="adni_regularizer_uai"
#for split in 0 1 2 3 4
#do
#  logdir="result/${experiment_name}"
#  python "${SRC}"/uai_main.py \
#          --logdir "$logdir" \
#          --fold "$split" \
#          --lr 3e-4 \
#          --blocks 2 2 2 2 \
#          --channels 16 16 32 64 \
#          --max_epochs 150 \
#          --use_uai
#
#  python "${SRC}"/run_adv_training.py \
#          --experiment_name "${experiment_name}_adv" \
#          --fold "${split}" \
#          --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#          --adv_num_epochs 200 \
#          --use_uai
#done