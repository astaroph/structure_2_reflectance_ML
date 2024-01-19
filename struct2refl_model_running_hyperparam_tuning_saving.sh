#!/bin/bash
#SBATCH -p  gpu # Partition to submit to
#SBATCH -n 1 # Number of cores
#SBATCH --gpus 1 # number of gpus to request
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1-05:00 # Runtime in days-hours:minutes
#SBATCH --mem 30000 # Memory in MB
#SBATCH -o /n/holyscratch01/pierce_lab/astaroph/scripts/script_output/struct2refl_model_running_hyperparam_tuning_saving%A.out # File to which standard out will be written
#SBATCH -e /n/holyscratch01/pierce_lab/astaroph/scripts/script_output/struct2refl_model_running_hyperparam_tuning_saving%A.err # File to which standard err will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=rchilders@g.harvard.edu # Email to which notifications will be sent
module load python
source activate /n/holylabs/LABS/pierce_lab/Users/astaroph/conda_envs/rocky8/pt2.0_cuda11.7_rocky8
module load gcc/12.2.0-fasrc01
cd ~/WingsAndWavelengths/

python struct2refl_model_running_hyperparam_tuning_saving.py -f ${1} \
-b ${2} \
-l ${3} \
-e ${4} \
-w ${5} \
-s ${6}
