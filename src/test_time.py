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
from compare_utils import *
from agents import *

import copy
import numpy as np
import math
import numpy.linalg as npla
import scipy.linalg as spla
import pickle
#import pandas as pd
#import plotnine as gg
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    #https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    #time=float(sys.argv[1])
    T=int(sys.argv[1])
    name=sys.argv[2]
    seed=int(sys.argv[3])
    agents = [make_sagald_agent(0.01), 
              make_prec_sagald_agent(0.01),
              make_sagald_agent(0.1),
              make_prec_sagald_agent(0.1)] 
    """agents = [make_mala_agent(),
          make_laplace_agent(),
          make_pgts_agent(),
          make_langevin_agent(),
          make_sagald_agent()]"""
    simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=1)
    agents_info = [(agents[0].theta, agents[0].gradient, agents[0].gradients),
                   (agents[1].theta, agents[1].gradient, agents[1].gradients, agents[1].weights),
                   (agents[2].theta, agents[2].gradient, agents[2].gradients),
                   (agents[3].theta, agents[3].gradient, agents[3].gradients, agents[3].weights)]
    pickle.dump(agents_info, open('%s_agents.p' % name,'wb'))
    data = (agents[0].contexts,agents[0].rewards)
    pickle.dump(data, open('%s_data.p' % name,'wb'))        


    # '../outputs/ma1.p'