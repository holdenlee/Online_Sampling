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
import numpy.linalg as npla
import scipy.linalg as spla
import pickle
#import pandas as pd
#import plotnine as gg


"""
num_articles = 1
dim = 20
dim1 = dim+1
sparsity = 5.0
theta_mean = 0
theta_std = 1
#time_limit= 0.1
T=1000
verbosity=1
"""

# Use default settings for Laplace agent for now (would be nice to tune this too). Not time-restricted.
def make_agent(agent_name, num_articles, dim, sparsity, t, verbosity=1, batch_size=64, bias_term=True):
    dim1 = dim+1 if bias_term else dim
    theta_mean = 0
    theta_std = 1
    epsilon1 = 0.01
    epsilon2 = 0.05
    alpha=0.2
    beta=0.5
    tol=0.0001
    if agent_name == "laplace":
        return LaplaceTSLogisticBandit(num_articles, dim1, theta_mean, theta_std, epsilon1,
                                                          alpha,beta,tol, verbosity=verbosity)
    elif agent_name == "online_laplace":
        ## Online version
        return OnlineDiagLaplaceTS(num_articles,dim1, [0]*dim1,
                                                            cov=None,init_pt=None,time=False,verbosity=verbosity)
    elif agent_name == "pg":
        # PG-TS has no parameters to tune
        return PGTS_Stream(num_articles, dim, intercept=bias_term, context_has_constant=bias_term, 
                                          n_steps=9999, time = t, verbosity=verbosity)
    elif agent_name == "mala":
        #self, num_articles, dim, intercept=False, context_has_constant=False, time=False, n_steps=100, verbosity=0
        # Langevin-based
        return MalaTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                     step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=9999,
                                     time = t,
                                         init_pt=None, verbosity=verbosity)
    elif agent_name == "mala_lf":
        #self, num_articles, dim, intercept=False, context_has_constant=False, time=False, n_steps=100, verbosity=0
        # Langevin-based
        return MalaTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                     step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=9999,
                                     time = t,
                                         init_pt=None, verbosity=verbosity, leapfrog = True)
    elif agent_name == "mala_untimed":
        ## Untimed MALA agent is used as baseline.
        ## I don't know whether I should use 1+t or 1+\sqrt{sparsity/dim}*t
        return MalaTS(num_articles, dim1, [0]*dim1, cov=None, 
                                             step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=500,
                                             init_pt=None, verbosity=verbosity)
    elif agent_name == "mala_lf_untimed":
        ## Untimed MALA agent is used as baseline.
        ## I don't know whether I should use 1+t or 1+\sqrt{sparsity/dim}*t
        return MalaTS(num_articles, dim1, [0]*dim1, cov=None, 
                                             step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=500,
                                             init_pt=None, verbosity=verbosity, leapfrog = True)
    elif agent_name == "sgld":
        ## With stochastic gradients
        return SGLDTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                            step_size=lambda t: 0.01/(1 + t * np.sqrt(sparsity/dim)),
                                            batch_size = batch_size,
                                            time=t,
                                            n_steps=9999,
                                            init_pt=None, verbosity=verbosity)
    elif agent_name == "sagald":
        return SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.05/(1 + t * np.sqrt(sparsity/dim)),
                                              batch_size = batch_size,
                                              time=t,
                                              n_steps=9999,
                                              init_pt=None, verbosity=verbosity)
    elif agent_name == "langevin":
        return BasicLangevinTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                                  step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=9999,
                                                  time = t,
                                                  init_pt=None, verbosity=verbosity)
    elif agent_name == "prec_sagald_nowt":
        return SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.05,
                                              batch_size = batch_size,
                                              time=t,
                                              n_steps=9999,
                                              precondition='proper',
                                              init_pt=None, verbosity=verbosity, weights=False)
    elif agent_name == "prec_sagald":
        return SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.05,
                                              batch_size = batch_size,
                                              time=t,
                                              n_steps=9999,
                                              precondition='proper',
                                              init_pt=None, verbosity=verbosity, weights=True)
    elif agent_name == "prec_sagald_cum":
        return SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.05,
                                              batch_size = batch_size,
                                              time=t,
                                              n_steps=9999,
                                              precondition='cum',
                                              init_pt=None, verbosity=verbosity)
    else:
        return None

def make_agents(agent_names, num_articles, dim, sparsity, t, verbosity=1, batch_size=64, bias_term=True):
    return [make_agent(name, num_articles, dim, sparsity, t, verbosity=verbosity, batch_size=batch_size, bias_term=bias_term) for name in agent_names]

def make_default_agents(num_articles, dim, sparsity, t, verbosity=1, batch_size=64, bias_term=True):
    return make_agents(['mala_lf',
                        'laplace',
                        'online_laplace',
                        'pg',
                        'langevin',
                        'sgld',
                        'sagald',
                        'prec_sagald_nowt',
                        'prec_sagald'], num_articles, dim, sparsity, t, verbosity=verbosity, batch_size=batch_size, bias_term=bias_term)

def make_shortlist_agents(num_articles, dim, sparsity, t, verbosity=1, batch_size=64, bias_term=True):
    return make_agents(['online_laplace',
                        'pg',
                        'sgld',
                        'sagald',
                        'prec_sagald'], num_articles, dim, sparsity, t, verbosity=verbosity, batch_size=batch_size, bias_term=bias_term)
