#!/bin/bash  

mkdir outputs/compare_time_0_1_trials_1000/
sbatch --array=1-100 -o outputs/compare_time_0_1_trials_1000/seed_%a.out compare.cmd 