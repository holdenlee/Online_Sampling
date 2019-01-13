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
    time=float(sys.argv[1])
    T=int(sys.argv[2])
    name=sys.argv[3]
    seed=int(sys.argv[4])
    agents = [make_mala_agent(time),
          make_laplace_agent(),
          make_online_laplace_agent(),
          make_pgts_agent(time),
          make_langevin_agent(time),
          make_sgld_agent(time),
          make_sagald_agent(time), #6
          make_prec_sagald_agent_nowt(time), #7
          make_prec_sagald_agent(time)] #8
    """agents = [make_mala_agent(),
          make_laplace_agent(),
          make_pgts_agent(),
          make_langevin_agent(),
          make_sagald_agent()]"""
    simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=1, force=True)
    agents_info = [agents[0].theta,
                   (agents[1].current_map_estimate, agents[1].current_Hessian),
                   (agents[2].est_coeffs, agents[2].est_inv_vars),
                   agents[3].theta,
                   agents[4].theta,
                   agents[5].theta,
                   (agents[6].theta, agents[6].gradient, agents[6].gradients),
                   (agents[7].theta, agents[7].gradient, agents[7].gradients),
                   (agents[8].theta, agents[8].gradient, agents[8].gradients, agents[8].weights)]
    pickle.dump(agents_info, open('%s_agents.p' % name,'wb'))
    data = (agents[0].contexts,agents[0].rewards)
    pickle.dump(data, open('%s_data.p' % name,'wb'))        
    samples_list = []
    agent0 = copy.deepcopy(agents[0])
    agent0.v=1
    agent0.time=0
    agent0.n_steps= 500
    agents = [agent0] + agents
    for agent in agents:
        samples = redraw_samples(agent, 1000, 2)
        samples_list += [samples]
    samples_list = np.asarray(samples_list)
    pickle.dump(samples_list, open('%s.p' % name,'wb'))
    print(merr(samples_list, 0.25))

    # '../outputs/ma1.p'