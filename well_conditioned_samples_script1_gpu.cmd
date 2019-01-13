#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:4

cd /u/holdenl/code/ts_tutorial/src
which python
pip install --user --upgrade pip
pip install --user pypolyagamma numpy scipy pandas
python well_conditioned_samples.py 0.01 1000 ../outputs/well_conditioned_samples_gpu_time_0_01_trials_1000 1

# #SBATCH --gres=gpu:1
# #SBATCH --mail-type=begin  
# #SBATCH --mail-type=end  
# #SBATCH --mail-user=holdenl@princeton.edu  
