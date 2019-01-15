#!/bin/bash

nohup python src/compare_later.py 0.1 1000 outputs/compare_later_time_0_1_trials_1000/seed 100 > outputs/compare_later_time_0_1_trials_1000/log.log 2>&1 &