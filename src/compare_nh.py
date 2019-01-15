from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys

from base.experiment import *
from logistic.env_logistic import *
from logistic.agent_logistic import *
from utils import *

import copy
import numpy as np
import math
#import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
#import pandas as pd
#import plotnine as gg
import pickle
import warnings
warnings.filterwarnings('ignore')

from agents import *
from compare_utils import *
#from graph_utils import *

time_limit=float(sys.argv[1])
T=int(sys.argv[2])
name=sys.argv[3]
seeds=int(sys.argv[4])

num_articles = 100
dim = 20
dim1= dim+1
sparsity = 5.0
theta_mean = 0
theta_std = 1

verbosity=1

batch_size = 64

for seed in range(2,seeds+1):
    agents = make_default_agents(num_articles, dim, sparsity, time_limit, verbosity=verbosity, batch_size=batch_size, bias_term=True)
    results, cum_regrets = simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=verbosity, dist_type='Bernoulli', slurm=True)
    print(results)
    print(cum_regrets)
    pickle.dump(results[0], open('%s_%d_results.p' % (name, seed),'wb'))
    pickle.dump(cum_regrets[0], open('%s_%d_regrets.p' % (name, seed),'wb'))
    

#run:
#nohup python compare_nh.py 0.1 1000 ../outputs/compare_time_0_1_trials_1000/seed 100 > ../outputs/compare_time_0_1_trials_1000/log.log 2>&1 &
"""
[30.112570676021964, 55.61220382233826, 44.82029246480474, 34.71447869419428, 35.94746494602103, 100.33048996420474, 41.95934741021247, 33.96480808424565, 39.080878341643015]
"""



