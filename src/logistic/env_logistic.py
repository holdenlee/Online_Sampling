''' News Article Recommendation'''

from __future__ import division
from __future__ import generators
from __future__ import print_function

import numpy as np

from base.environment import Environment
from base.distribution import *
from utils import *

class LogisticBandit(Environment):
  """Logistic Bandit Environment. The environment provides the features 
  vectors at any period and determines the rewards of a played action."""

  def __init__(self,num_articles,dim,theta_dist=None,arm_dist=None,seed=None,verbosity=0):
    #theta_mean=0,theta_std=1):
    # dim is actually (dimension of feature space) + 1 because it includes bias term
    """Args:
      num_articles - number of arms
      dim - dimension of the problem
      theta_dist - distribution of theta
      arm_dist - distribution of arms
      """
    Environment.__init__(self, verbosity)
    if seed is not None:
        np.random.seed(seed) #todo? upgrade this to its own RNG
    self.num_articles = num_articles
    self.dim = dim
    self.theta_dist = theta_dist if theta_dist != None else NormalDist(0,1,dim=dim)
    self.arm_dist = arm_dist if arm_dist != None else DistributionWithConstant(BernoulliDist(1.0/(dim - 1),dim - 1))
    #(lambda: np.random.binomial(1,max(0,1/(self.dim-1)),self.dim))
    #print(self.theta_dist)
    #print(type(self.theta_dist))
    self.theta = self.theta_dist()
    print('theta', self.theta)
    # keeping current rewards
    self.current_rewards = [0]*self.num_articles
    self.stochastic_rewards = [0]*self.num_articles
    
  def get_observation(self):
    '''generates context vector and computes the true
    reward of each article.'''
    
    context = []
    for i in range(self.num_articles):
      context_vector = self.arm_dist()
      context.append(context_vector)
      self.current_rewards[i] = 1/(1+np.exp(-self.theta.dot(context_vector)))
      self.stochastic_rewards[i] = np.random.binomial(1,self.current_rewards[i])
        
    return context
    
  def get_optimal_reward(self):
    return np.max(self.current_rewards)
    # return np.max(self.stochastic_rewards)
  
  def get_expected_reward(self,article):
    return self.current_rewards[article]
  
  def get_stochastic_reward(self,article):
    #expected_reward = self.get_expected_reward(article)
    #stochastic_reward = np.random.binomial(1,expected_reward)
    stochastic_reward = self.stochastic_rewards[article]
    return stochastic_reward

class FixedLogisticBandit(LogisticBandit):
  def __init__(self,num_articles,dim,theta_dist=None,arm_dist=None,seed=None,verbosity=0):
    LogisticBandit.__init__( self,num_articles,dim,theta_dist=theta_dist,arm_dist=arm_dist,seed=seed,verbosity=verbosity)
    self.context_vectors = [self.arm_dist() for _ in range(self.num_articles)]
  def get_observation(self):
    '''generates context vector and computes the true
    reward of each article.'''
    for i in range(self.num_articles):
      self.current_rewards[i] = 1/(1+np.exp(-self.theta.dot(self.context_vectors[i])))
      self.stochastic_rewards[i] = np.random.binomial(1,self.current_rewards[i])
    return self.context_vectors
