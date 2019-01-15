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

from utils import *

import copy
import numpy as np
import math
import numpy.linalg as npla
import scipy.linalg as spla
import pickle
#import pandas as pd
#import plotnine as gg


def round_down(val, amt):
    return amt*math.floor(val/amt)

def round_up(val, amt):
    return amt*math.ceil(val/amt)

def intervals(xmin,xmax,spacing):
    return [xmin + spacing*i for i in range(int((xmax-xmin)/spacing)+1)]

def make_hist(a, bin_length):
    a_min = np.min(a)
    a_max = np.max(a)
    # round down bin length
    hist, edges = np.histogram(a, intervals(round_down(a_min, bin_length),
                                round_up(a_max, bin_length),
                                bin_length))
    return hist, edges

def make_hist_from_std_dev(a,factor):
    s = np.std(a)
    bin_length = factor*s
    return make_hist(a, bin_length)

def make_hists(arrs, bin_length):
    a_min = np.min(arrs)
    a_max = np.max(arrs)
    # round down bin length
    li = [np.histogram(a, intervals(round_down(a_min, bin_length),
                                round_up(a_max, bin_length),
                                bin_length)) for a in arrs]
    edges = li[0][1]
    hists = [t[0] for t in li] 
    return hists, edges

def make_hists_from_std_dev(arrs, factor):
    s = np.std(arrs[0])
    bin_length = factor*s
    return make_hists(arrs, bin_length)

def tv(a1,a2):
    return np.sum(np.abs((a1-a2).astype(float)))/(2*np.sum(a1))

#first array is ground truth
def merr(arrs, factor,verbosity=1):
    (num_arrs, sample_size, d) = arrs.shape
    tvs = np.zeros((num_arrs-1, d))
    hists_list = []
    for i in range(d):
        arr_i = arrs[:,:,i]
        hists, edges = make_hists_from_std_dev(arr_i, factor)
        printv(hists, verbosity,1)
        hists_list += [hists]
        for j in range(num_arrs-1):
            tvs[j,i] = tv(hists[0], hists[j+1])
    merrs = np.sum(tvs, axis=1)/d
    return tvs, merrs, hists
#def ma(true_samples, samples, side_length):


def redraw_samples(agent, num, verbosity=1):
    samples = []
    #theta = agent.theta
    agent_copy = copy.deepcopy(agent)
    for i in range(num):
        printv("Drawing sample %d" % (i+1), verbosity, 1)
        #agent_copy.theta= theta
        s = agent_copy.get_sample()
        samples += [s]
        printv(" "+repr(s), verbosity, 2)
    return np.asarray(samples)
#def redraw_samples_sagald(agent, num, verbosity=1):

def redraw_samples_for_agents(agents, num, verbosity=1):
    samples_list = []

    for agent in agents:
        samples = redraw_samples(agent, num, verbosity)
        samples_list += [samples]
    samples_list = np.asarray(samples_list)
    return samples_list

def simple_compare(agents, num_articles, dim, sparsity, n_steps, seed=0, verbosity=0, force=False, graph=False, dist_type='Bernoulli', toggle_at=0, slurm=False):
    #env = FixedLogisticBandit(num_articles, dim, DistributionWithConstant(NormalDist(0,1,dim=dim-1),-2.5), DistributionWithConstant(BernoulliDist(5.0/(dim-1),dim-1)), seed=seed)
    env = LogisticBandit(num_articles, dim+1, NormalDist(0,1,dim=dim+1), DistributionWithConstant(BernoulliDist(sparsity/dim,dim)), seed=seed) if dist_type=='Bernoulli' else LogisticBandit(num_articles, dim, NormalDist(0,1,dim=dim), (NormalDist(0,sparsity,dim=dim)), seed=seed) #'Normal'
    #sparsity is var in the second case. #rn, Bernoulli is no-bias.
    experiment = ExperimentCompare(agents, env, n_steps,
                   seed=seed, verbosity=verbosity, force=force, toggle_at=toggle_at, slurm=slurm)
    experiment.run_experiment()
    results = []
    results.append(experiment.results)
    cum_regrets = experiment.cum_regret
    #if graph:
    #    plot_results(results)
    return results, cum_regrets

def simple_compares(make_agents, num_articles, dim, sparsity, n_steps, seeds, verbosity=0, graph=False, dist_type='Bernoulli'):
    results_list = []
    cum_regrets_list = []
    avg_regrets = np.zeros(len(make_agents))
    for seed in seeds:
        agents = [make_agent() for make_agent in make_agents]
        results, cum_regrets = simple_compare(agents, num_articles, dim, sparsity, n_steps, seed=seed, verbosity=verbosity, graph=graph, dist_type=dist_type)
        results_list += [results]
        cum_regrets_list += [cum_regrets]
        avg_regrets += cum_regrets
    avg_regrets = avg_regrets / len(seeds)
    return results_list, cum_regrets_list, avg_regrets

def hyperparameter_sweep(make_agent_from_hyperparameters, hyperparam_list, dim, sparsity, n_steps, seed=0, verbosity=0, graph=False):
    agents = [make_agent_from_hyperparameters(hyperparam) for hyperparam in hyperparam_list]
    results, cum_regrets = simple_compare(agents, num_articles, dim, sparsity, n_steps, seed=seed, verbosity=verbosity, graph=graph)
    