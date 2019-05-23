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
#import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import pickle
import pandas as pd
#import plotnine as gg
import warnings
warnings.filterwarnings('ignore')


#https://www.tutorialspoint.com/python/python_command_line_arguments.htm
#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
time=0.1
T=1000
name='outputs/ma/wc'
#seed=201
num_articles = 1
dim = 20
dim1 = dim+1
sparsity = 5.0
verbosity=1
batch_size = 64

for seed in range(1001,1011):
	agents = make_shortlist_agents(num_articles, dim, sparsity, time, verbosity=verbosity, batch_size=batch_size, bias_term=True)
	mala_agent = MalaTS(num_articles, dim1, [0]*dim1, cov=None, 
												 step_size=lambda t: 0.5/(1 + t * np.sqrt(sparsity/dim)), time=time, n_steps=9999,
												 init_pt=None, verbosity=verbosity, leapfrog = True)
	"""['mala_lf',
						'online_laplace',
						'pg',
						'sgld',
						'sagald',
						'prec_sagald']"""
	agents = [mala_agent]+agents
	results = simple_compare(agents, num_articles, dim, sparsity, T, seed, verbosity=1, force=True)
	agents_info =\
		[agents[0].theta,
				   (agents[1].est_coeffs, agents[1].est_inv_vars),
				   agents[3].theta,
				   (agents[4].theta, agents[4].gradient, agents[4].gradients),
				   (agents[5].theta, agents[5].gradient, agents[5].gradients, agents[5].weights)] 
	#for i in range(6):
	#    (agents[i].contexts,agents[i].rewards) = (contexts, rewards)
	pickle.dump(agents_info, open('%s_%d_agents.p' % (name, seed),'wb'))
	data = (agents[0].contexts,agents[0].rewards)
	pickle.dump(data, open('%s_%d_data.p' % (name, seed),'wb'))        
	samples_list = []
	agent0 = copy.deepcopy(agents[0])
	agent0.v=1
	agent0.time=0
	agent0.n_steps=1000
	agents = [agent0] + agents
	i=0
	for agent in agents:
		samples = redraw_samples(agent, 1000, 2)
		pickle.dump(samples, open('%s_%d_%d.p' % (name,seed,i),'wb'))
		samples_list += [samples]
		i+=1
	samples_list = np.asarray(samples_list)
	pickle.dump(samples_list, open('%s_%d.p' % (name,seed),'wb'))
	print(merr(samples_list, 0.25))

	# '../outputs/ma1.p'