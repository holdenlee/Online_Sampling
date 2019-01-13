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

num_articles = 1
dim = 20
dim1 = dim+1
sparsity = 5.0
theta_mean = 0
theta_std = 1
#time_limit= 0.1
T=1000
verbosity=1

# Use default settings for Laplace agent for now (would be nice to tune this too). Not time-restricted.

epsilon1 = 0.01
epsilon2 = 0.05
alpha=0.2
beta=0.5
tol=0.0001
make_laplace_agent = lambda: LaplaceTSLogisticBandit(num_articles, dim1, theta_mean, theta_std, epsilon1,
                                                      alpha,beta,tol, verbosity=verbosity)

## Online version

make_online_laplace_agent = lambda: OnlineDiagLaplaceTS(num_articles,dim1, [0]*dim1,
                                                        cov=None,init_pt=None,time=False,verbosity=0)

# PG-TS has no parameters to tune

make_pgts_agent = lambda t: PGTS_Stream(num_articles, dim, intercept=True, context_has_constant=True, 
                                      n_steps=9999, time = t, verbosity=verbosity)
#self, num_articles, dim, intercept=False, context_has_constant=False, time=False, n_steps=100, verbosity=0

# Langevin-based

make_mala_agent = lambda t: MalaTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                 step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=9999,
                                 time = t,
                                     init_pt=None, verbosity=verbosity)

## Untimed MALA agent is used as baseline.
## I don't know whether I should use 1+t or 1+\sqrt{sparsity/dim}*t

make_mala_agent_untimed = lambda s: MalaTS(num_articles, dim1, [0]*dim1, cov=None, 
                                         step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=s,
                                         init_pt=None, verbosity=verbosity)

## With stochastic gradients


make_sgld_agent = lambda t: SGLDTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                        step_size=lambda t: 0.01/(1 + t * np.sqrt(sparsity/dim)),
                                        batch_size = 64,
                                        time=t,
                                        n_steps=9999,
                                        init_pt=None, verbosity=verbosity)
make_sagald_agent = lambda t: SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                          step_size=lambda t: 0.05/(1 + t * np.sqrt(sparsity/dim)),
                                          batch_size = 64,
                                          time=t,
                                          n_steps=9999,
                                          init_pt=None, verbosity=verbosity)
make_langevin_agent = lambda t: BasicLangevinTS(num_articles, dim1, [0]*(dim1), cov=None, 
                                              step_size=lambda t: 0.1/(1 + t * np.sqrt(sparsity/dim)), n_steps=9999,
                                              time = t,
                                              init_pt=None, verbosity=verbosity)

make_prec_sagald_agent_nowt = lambda t: SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                          step_size=lambda t: 0.05,
                                          batch_size = 64,
                                          time=t,
                                          n_steps=9999,
                                          precondition='proper',
                                          init_pt=None, verbosity=verbosity, weights=False)
make_prec_sagald_agent = lambda t: SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                          step_size=lambda t: 0.05,
                                          batch_size = 64,
                                          time=t,
                                          n_steps=9999,
                                          precondition='proper',
                                          init_pt=None, verbosity=verbosity, weights=True)
make_prec_cum_sagald_agent = lambda t: SAGATS(num_articles, dim1, [0]*(dim1), cov=None, 
                                          step_size=lambda t: 0.05,
                                          batch_size = 64,
                                          time=t,
                                          n_steps=9999,
                                          precondition='cum',
                                          init_pt=None, verbosity=verbosity)