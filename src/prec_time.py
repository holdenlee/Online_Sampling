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
seed=int(sys.argv[4])

num_articles = 100
dim = 20
dim1= dim+1
sparsity = 5.0
theta_mean = 0
theta_std = 1

verbosity=1

batch_size = 64

def make_sagald(n_steps):
    return SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.05,
                                              batch_size = batch_size,
                                              time=0,
                                              n_steps=n_steps,
                                              precondition='proper',
                                              init_pt=None, verbosity=verbosity, weights=True)

agents = [make_sagald(100)]
results, cum_regrets = simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=verbosity, dist_type='Bernoulli')
print(results)
print(cum_regrets)
pickle.dump((results, cum_regrets), open('%s.p' % name,'wb'))

