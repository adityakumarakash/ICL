#!/bin/bash

python src/compute_mean_std.py \
              --run_dir 'result/german_fairness/FC-German/runs/' \
              --run_regex 'icl.*' \
              --tags 'navib_acc/test,Navib_Adv3_acc/test'

#python src/compute_mean_std.py \
#              --run_dir 'result/adult_fairness/FC-Adult/runs/' \
#              --run_regex 'icl.*' \
#              --tags 'navib_acc/test,Navib_Adv3_acc/test'