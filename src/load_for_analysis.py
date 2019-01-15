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
#import plotnine as gg
import pickle
import warnings
warnings.filterwarnings('ignore')

from agents import *
from compare_utils import *
#from graph_utils import *

name = '../outputs/compare_0_1_trials_1000_seed'
results_list = []
regrets_list = []
for i in range(1,101):
    results, regrets = pickle.load(open('%s_%d.p' % (name, i), 'rb'))
    pickle.dump(results[0], open('%s_results_%d.p' % (name, i), 'wb'))
    pickle.dump(regrets, open('%s_regrets_%d.p' % (name, i), 'wb'))
