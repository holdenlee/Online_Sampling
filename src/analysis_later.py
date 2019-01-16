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

name = '../outputs/compare_later_time_0_1_trials_1000/seed'
results_list = []
regrets_list = []
for i in range(1,19):
    results = pd.read_pickle(open('%s_%d_results.p' % (name, i), 'rb'))
    regrets = pickle.load(open('%s_%d_regrets.p' % (name, i), 'rb')) if i != 1 else [9.609577211676948, 9.631305721679768, 9.595549955459072, 9.049107889173861, 9.594515897813, 23.045289883872908, 9.199815752495159, 9.180508408854028, 9.264632848194701]
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

"""
[ 49.24954836  47.65126672  51.82073288  49.37682708  48.72826463
 164.45889434  49.07907157  48.97757667  49.4270332 ]
"""

