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

time_limit= 0.1 #float(sys.argv[1])
T= 2000 #int(sys.argv[2])
toggle_at = 1000
name= 'outputs/compare_later_time_0_1_trials_1000/seed'#sys.argv[3]
seeds = 100 #int(sys.argv[4])

num_articles = 100
dim = 20
dim1= dim+1
sparsity = 5.0
theta_mean = 0
theta_std = 1

verbosity=1

batch_size = 64

for seed in range(1,seeds+1):
    agents = make_default_agents(num_articles, dim, sparsity, time_limit, verbosity=verbosity, batch_size=batch_size, bias_term=True)
    results, cum_regrets = simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=verbosity, dist_type='Bernoulli', toggle_at=toggle_at)
    print(results)
    print(cum_regrets)
    pickle.dump(results, open('%s_%d_results.p' % (name, seed),'wb'))
    pickle.dump(cum_regrets, open('%s_%d_regrets.p' % (name, seed),'wb'))

    """
    [9.609577211676948, 9.631305721679768, 9.595549955459072, 9.049107889173861, 9.594515897813, 23.045289883872908, 9.199815752495159, 9.180508408854028, 9.264632848194701]
    """
