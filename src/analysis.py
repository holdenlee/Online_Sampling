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
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import pandas as pd
import plotnine as gg
import pickle
import warnings
warnings.filterwarnings('ignore')

from agents import *
from compare_utils import *
from graph_utils import *

name = '../outputs/compare_time_0_1_trials_1000/seed'
results_list = []
regrets_list = []
for i in range(1,34):
    results = pd.read_pickle(open('%s_%d_results.p' % (name, i), 'rb'))
    regrets = pickle.load(open('%s_%d_regrets.p' % (name, i), 'rb')) if i != 1 else [30.112570676021964, 55.61220382233826, 44.82029246480474, 34.71447869419428, 35.94746494602103, 100.33048996420474, 41.95934741021247, 33.96480808424565, 39.080878341643015]
    results_list += [results]
    regrets_list += [regrets]
regrets = np.asarray(regrets_list)
avg_regrets = np.mean(regrets, axis=0)
print(['mala_lf',
                        'laplace',
                        'online_laplace',
                        'pg',
                        'langevin',
                        'sgld',
                        'sagald',
                        'prec_sagald_nowt',
                        'prec_sagald'])
print(avg_regrets)

"""34
[43.4844211  44.25713725 43.93010964 43.62387286 43.66123608 45.61223683
 43.84341433 43.60115556 43.756188  ]

"""

