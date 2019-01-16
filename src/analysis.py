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
for i in range(15):
    results = pd.read_pickle(open('%s_%d_results.p' % (name, i), 'rb'))
    regrets = pickle.load(open('%s_%d_regrets.p' % (name, i), 'rb'))
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