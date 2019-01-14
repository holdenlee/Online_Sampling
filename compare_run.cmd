#!/bin/bash  

sbatch --array=1-100 -o outputs/compare_time_0_1_trials_1000_seed_%a.out compare.cmd 