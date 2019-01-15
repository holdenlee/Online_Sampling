#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1

cd /u/holdenl/code/ts_tutorial/src
which python
pip install --user --upgrade pip
pip install --user pypolyagamma numpy scipy pandas
python load_for_analysis.py

# #SBATCH --gres=gpu:1
# #SBATCH --mail-type=begin  
# #SBATCH --mail-type=end  
# #SBATCH --mail-user=holdenl@princeton.edu  
