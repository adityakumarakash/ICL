# Learning Invariant Representations using ICL
We provide code for all the experiments done
using ICL loss. The organization of code is as follows :
- All code is present in ``src`` directory. 
The code for ADNI classification is present in `src/adni_classification`
- Bash files required to run the experiments with all 
the required hyperparameters are in `bin` directory.

### Requirements 
We attach the conda `requirements.txt` file to provide 
exact environment details used for running the experiments. 

The main dependencies are 
- pytorch
- tensorflow for tensorboard
- PIL 
- sklearn
- numpy
- torchvision
- MulticoreTSNE
- matplotlib
- tqdm


## Running Experiments
Next, we provide details on how to run each experiment one by one.  
The command to run each baseline with the hyperparameters
used for reporting the results
are present in relevant `bin/<experiment-filename>.sh` file
along with the random seeds.   
Most of the code, commands and argument names are self-explanatory.

### MNIST Style Experiment
The relevant code is in : `src/mnist_style_exp.py`. 

Running the experiment :
1) Check `bin/mnist_style_exp.sh`. Each baseline is 
run for 10 random seeds.
To run a baseline uncomment the appropriate section
of the file.
2) Note that `comp_lambda_max` is the hyperparameter 
for regularization weight for this experiment.
3) Run ``bash bin/mnist_style_exp.sh`` with 
the selected baseline and parameters.
4) The results can be seen in `result/<experiment_name>/<model_name>-<dataset_name>/<run_id>` directory,
where `<run_id>` is a string consisting of all relevant 
parameters used for this experiment run.
5) The experiment statistics are dumped in tensorboard. Check using
`tensorboard --logdir <logdir> --port 6006`.

The tSNE plot for this experiments can be found under the relevant
`<run_id>` directory.
Apart from visual inspection,
the mean and standard deviation of multiple runs can be obtained
using `bin/check_mean_std.sh`. Check this file to specify the
relevant regex for the run and the appropriate tags
to extract required scores.

### Fairness Dataset Experiments
The relevant code is in: `src/fairness_exp.py`

The data required to run this experiment is already provided
in `data/` directory. The train, validation and test 
splits used for reporting results are already
present in this directory. To generate fresh splits
use `src/uci_data.py`.

Running the experiments:
1. Check `bin/fairness_exp_adult.sh` for running experiments
for Adult dataset and `bin/fairness_exp_german.sh` for 
running experiments for German dataset
2. `alpha_max` is the hyperparameter 
for regularization weight for this experiment
3. Uncomment the relevant baseline command and run  
   a) `bash bin/fairness_exp_adult.sh` for Adult dataset and,   
   b) `bash bin/fairness_exp_german.sh` for German dataset

The results are dumped in the `result/<experiment_name>` directory. Follow
the instructions from MNIST style experiment to visualize 
the results.


### Continuous Extraneous Variable Experiment
The relevant code is in: `src/continuous_extraneous_exp.py`. 

The data required to run this experiment is already provided
in `data/` directory. The train, validation and test 
splits used for reporting results are already
present in this directory. To generate fresh splits
use `src/uci_age_protected.py`.

Running the experiments:
1. Check `bin/continuous_extraneous_exp.sh` for running experiments
2. `alpha_max` is the hyperparameter 
for regularization weight for this experiment.
3. Uncomment the relevant baseline command and run 
`bash bin/continuous_extraneous_exp.sh`

The results are dumped in the `result/<experiment_name>` directory. Follow
the instructions from MNIST style experiment to visualize 
the results.


### Rotation Invariance Experiment on MNIST-Rot

The relevant code is in: `src/mnist_rot_exp.py`. 

The data required to run this experiment is automatically
downloaded from MNIST dataset provided by torchvision package. 
The rotations associated with each image are
specified using a random sequence generated
from numpy with a fixed seed.

Running the experiments:
1. Check `bin/mnist_rot_exp.sh` for running experiments
2. `alpha_max` is the hyperparameter 
for regularization weight for this experiment.
3. Uncomment the relevant baseline command and run 
`bash bin/mnist_rot_exp.sh`

The results are dumped in the `result/<experiment_name>` directory. Follow
the instructions from MNIST style experiment to visualize 
he results.


### ADNI Experiments
The relevant code is in : `src/adni_classification` directory.

The random splits used for reporting the results
are provided in `data/splits` directory. 

Running the experiments:
1. Create a symbolic link to data using `ln -s <ADNI-data-dir> ./data/adni_data`
2. Check `bin/run_adni_classification.sh` for running experiments
3. `alpha_max` is the hyperparameter 
for regularization weight for this experiment.
4. Uncomment the relevant baseline command and run 
`bash bin/run_adni_classification.sh`
5. The results are dumped in `result/<experiment-name>/fold_<fold-number>/` directory.
The corresponding run of the adversary is present in 
`result/<experiment-name>_adv/` directory.

Visualize tensorboard same as instructions from MNIST style experiment.