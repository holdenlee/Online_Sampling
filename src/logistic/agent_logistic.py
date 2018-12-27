"""Agents for logistic problem."""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import random as rnd

from algorithms.langevin import *
from base.agent import Agent
#from base.distribution import *
from base.timing import *
import time
import sys

_SMALL_NUMBER = 1e-10
_MEDIUM_NUMBER=.01
_LARGE_NUMBER = 1e+2
##############################################################################

class GreedyLogisticBandit(Agent):
  """Greedy News Recommender."""
  
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001,time=True):
    """Args:
      num_articles - number of news articles
      dim - dimension of the problem
      theta_mean - mean of each component of theta
      theta_std - std of each component of theta
      epsilon - used in epsilon-greedy.
      alpha - used in backtracking line search
      beta - used in backtracking line search
      tol - stopping criterion of Newton's method.
      """
      
    self.num_articles = num_articles
    self.dim = dim
    self.theta_mean = theta_mean
    self.theta_std = theta_std
    self.back_track_alpha = alpha
    self.back_track_beta = beta
    self.tol = tol
    self.epsilon = epsilon
    self.time = time
   
    # keeping current map estimates and Hessians for each news article
    self.current_map_estimate = self.theta_mean*np.ones(self.dim)
    #[self.theta_mean*np.ones(self.dim) 
    #                                        for _ in range(self.num_articles)]
    self.current_Hessian = np.diag([1/self.theta_std**2]*self.dim)
                                            #for _ in range(self.num_articles)]
  
    # keeping the observations for each article
    self.num_plays = 0
      #[0 for _ in range(self.num_articles)]
    self.contexts = np.zeros((0,self.dim))
    #self.contexts = np.asarray([])
      #[[] for _ in range(self.num_articles)]
    self.rewards = np.asarray([]) 
      #[[] for _ in range(self.num_articles)]
    
  def _compute_gradient_hessian_prior(self,x):
    '''computes the gradient and Hessian of the prior part of 
        negative log-likelihood at x.'''
    Sinv = np.diag([1/self.theta_std**2]*self.dim) 
    mu = self.theta_mean*np.ones(self.dim)
    
    g = Sinv.dot(x - mu)
    H = Sinv
    
    return g,H
  
  def _compute_gradient_hessian(self,x): #,article):
    """computes gradient and Hessian of negative log-likelihood  
    at point x, based on the observed data for the given article."""
    
    g,H = self._compute_gradient_hessian_prior(x)
    # this needs to be done with a matrix.
    zs = self.contexts.T
    ys = self.rewards
    preds = 1/(1+np.exp(-x.dot(zs)))
    #print(np.shape(zs),np.shape(ys),np.shape(x),np.shape(preds))
    g = zs.dot(preds - ys)
    H = H + zs.dot(np.diag(np.multiply(preds, 1-preds))).dot(zs.T)
    return g, H
    """
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      pred = 1/(1+np.exp(-x.dot(z)))
      
      g = g + (pred-y)*z
      H = H + pred*(1-pred)*np.outer(z,z)
    
    return g,H
    """

  def _evaluate_log1pexp(self, x):
    """given the input x, returns log(1+exp(x))."""
    return np.piecewise(x, [x>_LARGE_NUMBER], [lambda x: x, lambda x: np.log(1+np.exp(x))])
    """
    if x > _LARGE_NUMBER:
      return x
    else:
      return np.log(1+np.exp(x))
    """

  def _evaluate_negative_log_prior(self, x):
    """returning negative log-prior evaluated at x."""
    Sinv = np.diag([1/self.theta_std**2]*self.dim) 
    mu = self.theta_mean*np.ones(self.dim)
    
    return 0.5*(x-mu).T.dot(Sinv.dot(x-mu))

  def _evaluate_negative_log_posterior(self, x): #, article):
    """evaluate negative log-posterior for article at point x."""

    value = self._evaluate_negative_log_prior(x)
    zs = self.contexts.T
    ys = self.rewards
    return (np.sum(self._evaluate_log1pexp(x.dot(zs))) - x.dot(zs).dot(ys)) #y.T?
    """
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      value = value + self._evaluate_log1pexp(x.dot(z)) - y*x.dot(z)
      
    return value
    """
  
  def _back_track_search(self, x, g, dx): #, article):
    """Finding the right step size to be used in Newton's method.
    Inputs:
      x - current point
      g - gradient of the function at x
      dx - the descent direction

    Retruns:
      t - the step size"""

    step = 1
    current_function_value = self._evaluate_negative_log_posterior(x) #, article)
    difference = self._evaluate_negative_log_posterior(x + step*dx) - \
        (current_function_value + self.back_track_alpha*step*g.dot(dx))
    while difference > 0:
      step = self.back_track_beta * step
      difference = self._evaluate_negative_log_posterior(x + step*dx) - \
          (current_function_value + self.back_track_alpha*step*g.dot(dx))

    return step

  def _optimize_Newton_method(self): #, article):
    """Optimize negative log_posterior function via Newton's method for the
    given article."""
    
    x = self.current_map_estimate #s[article]
    error = self.tol + 1
    while error > self.tol:
      g, H = self._compute_gradient_hessian(x) #,article)
      delta_x = -npla.solve(H, g)
      step = self._back_track_search(x, g, delta_x) #, article)
      x = x + step * delta_x
      error = -g.dot(delta_x)
      #print(error, self.tol)
      sys.stdout.write('.')
      
    # computing the gradient and hessian at final point
    g, H = self._compute_gradient_hessian(x) #,article)

    # updating current map and Hessian for this article
    self.current_map_estimate = x #s[article] = x
    self.current_Hessian = H #s[article] = H
    return x, H
  
  def update_observation(self, context, article, feedback):
    '''updates the observations for displayed article, given the context and 
    user's feedback. The new observations are saved in the history of the 
    displayed article and the current map estimate and Hessian of this 
    article are updated right away.
    
    Args:
      context - a list containing observed context vector for each article
      article - article which was recently shown
      feedback - user's response.
      '''
    if self.time:
      start = time.time()
    self.num_plays +=1
      #[article] += 1
    self.contexts = np.append(self.contexts,[context[article]], axis=0)
    self.rewards = np.append(self.rewards, [feedback])
    #self.contexts[article].append(context[article])
    #self.rewards[article].append(feedback)
    
    # updating the map estimate and Hessian for displayed article
    _,__ = self._optimize_Newton_method() #article)
    if self.time:
      end=time.time()
      print("(update_observation, %s) Time Elapsed: %f" % (type(self).__name__, end - start))
  
  def _map_rewards(self,context):
    map_rewards = []
    theta = self.current_map_estimate
    for i in range(self.num_articles):
      x = context[i]
      #theta = self.current_map_estimates[i]
      map_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return map_rewards
  
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    if self.time:
      start = time.time()
    map_rewards = self._map_rewards(context)
    article = np.argmax(map_rewards)
    if self.time:
      end=time.time()
      print("(pick_action, %s) Time Elapsed: %f" % (end - start))
    return article
##############################################################################
class EpsilonGreedyLogisticBandit(GreedyLogisticBandit):
  '''Epsilon greedy agent for the news recommendation problem.'''
  
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    if self.time: 
      start = time.time()
    map_rewards = self._map_rewards(context)
    if np.random.uniform()<self.epsilon:
      article = np.random.randint(0,self.num_articles)
    else:
      article = np.argmax(map_rewards)
    if self.time:
      end=time.time()
      print("(pick_action, %s) Time Elapsed: %f" % (type(self).__name__, end - start))
    return article

##############################################################################
class LaplaceTSLogisticBandit(GreedyLogisticBandit):   
  '''Laplace approximation to TS for news recommendation problem.'''
  def _sampled_rewards(self,context):
    sampled_rewards = []
    mean = self.current_map_estimate #s[i]
    cov = npla.inv(self.current_Hessian) #s[i])
    theta = np.random.multivariate_normal(mean, cov)
    for i in range(self.num_articles):
      x = context[i]
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    if self.time: 
      start = time.time()
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    if self.time:
      end=time.time()
      print("(pick_action, %s) Time Elapsed: %f" % (type(self).__name__, end - start))
    return article

##############################################################################
class LangevinTSLogisticBandit(GreedyLogisticBandit):
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001,batch_size = 100, step_count=200,
               step_size=.01,time=True):
    GreedyLogisticBandit.__init__(self,num_articles,dim,theta_mean,theta_std,
                                  epsilon,alpha,beta,tol,time)
    self.batch_size = batch_size
    self.step_count = step_count
    self.step_size = step_size
    self.x = np.zeros(dim)
    
  def _compute_stochastic_gradient(self, x): #, article):
    '''computes a stochastic gradient of the negative log-posterior for the given
     article.'''
    
    if self.num_plays <= self.batch_size: #[article]<=self.batch_size:
      sample_indices = range(self.num_plays) #[article])
      gradient_scale = 1
    else:
      gradient_scale = self.num_plays/self.batch_size #[article]/self.batch_size
      sample_indices = rnd.sample(range(self.num_plays),self.batch_size) #[article]),self.batch_size)
    
    zs = self.contexts[sample_indices].T
    ys = self.rewards[sample_indices]
    preds = 1/(1+np.exp(-x.dot(zs)))
    g = zs.dot(preds - ys)
    #g = preds.dot(ys)
        #ys.dot(preds.T)
    """
    g = np.zeros(self.dim)
    for i in sample_indices:
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      preds = 1/(1+np.exp(-x.dot(z)))
      g = g + (pred-y)*z
    """
    g_prior,_ = self._compute_gradient_hessian_prior(x) 
    g = gradient_scale*g + g_prior
    return g
  
  def _Langevin_samples(self):
    '''gives the Langevin samples for each of the articles'''
    # determining starting point and conditioner
    x = self.current_map_estimate
    preconditioner = npla.inv(self.current_Hessian)
    preconditioner_sqrt=spla.sqrtm(preconditioner)
      
    #Remove any complex component in preconditioner_sqrt arising from numerical error
    complex_part=np.imag(preconditioner)
    if (spla.norm(complex_part)> _SMALL_NUMBER):
      print("Warning. There may be numerical issues.  Preconditioner has complex values")
      print("Norm of the imaginary component is, ")+str(spla.norm(complex_part))
    preconditioner_sqrt=np.real(preconditioner_sqrt)
      
    for i in range(self.step_count):
      g = -self._compute_stochastic_gradient(x) #,a)
      scaled_grad=preconditioner.dot(g)
      scaled_noise = preconditioner_sqrt.dot(np.random.randn(self.dim)) 
      x = x + self.step_size * scaled_grad+np.sqrt(2*self.step_size)*scaled_noise
      sys.stdout.write('*')
    self.x=x
    return x
  
  def _sampled_rewards(self,context):
    sampled_rewards = []
    theta = self._Langevin_samples()
    for i in range(self.num_articles):
      x = context[i]
      #theta = sampled_theta[i]
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    if self.time: 
      start = time.time()
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    if self.time:
      end=time.time()
      print("(pick_action, %s) Time Elapsed: %f" % (type(self).__name__, end - start))
    return article
 
class SAGALDTSLogisticBandit(LangevinTSLogisticBandit):
  def _compute_stochastic_gradient(self, x):
    '''computes a stochastic gradient of the negative log-posterior for the given
     article.'''
    xstar = self.current_map_estimate #s[article]
    if self.num_plays<=self.batch_size:
      sample_indices = range(self.num_plays)
      gradient_scale = 1
    else:
      gradient_scale = self.num_plays/self.batch_size
      sample_indices = rnd.sample(range(self.num_plays),self.batch_size)
    
    zs = self.contexts[sample_indices].T
    ys = self.rewards[sample_indices]
    preds = 1/(1+np.exp(-x.dot(zs)))
    if self.num_plays <= self.batch_size:
        g = zs.dot(preds - ys)
    else:
        pred_xstar = 1/(1+np.exp(-xstar.dot(zs)))
        g = zs.dot(preds - pred_xstar) #variance-reduced gradient
        #g = (preds - pred_xstar).dot(preds.T) 
        
#     g = np.zeros(self.dim)
#     for i in sample_indices: #should parallelize this!
#       z = self.contexts[i]
#       y = self.rewards[i]
#       pred = 1/(1+np.exp(-x.dot(z)))
#       if self.num_plays[article]<=self.batch_size:
#         g = g + (pred-y)*z
#       else:
#         pred_xstar = 1/(1+np.exp(-xstar.dot(z)))
#         g = g + (pred - pred_xstar)*z #variance-reduced gradient
                        
    g_prior,_ = self._compute_gradient_hessian_prior(x)
    g = gradient_scale*g + g_prior
    return g
    
class OSAGALDTSLogisticBandit(LangevinTSLogisticBandit):
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001,batch_size = 100, step_count=200,
               step_size=.01,time=True):
    LangevinTSLogisticBandit.__init__(self,num_articles,dim,theta_mean,theta_std,
                                  epsilon,alpha,beta,tol,batch_size, step_count, step_size, time)
    #estimated gradients
    self.gradients = np.zeros((0,self.dim)) #asarray([])
    #sum of self.gradients
    self.gradient = np.zeros(self.dim)
    #self.x = np.zeros(self.dim)
  
  def _Langevin_samples(self):
    '''gives the Langevin samples for each of the articles'''
    # determining starting point and conditioner
    preconditioner = npla.inv(self.current_Hessian)
    preconditioner_sqrt=spla.sqrtm(preconditioner)
      
    #Remove any complex component in preconditioner_sqrt arising from numerical error
    complex_part=np.imag(preconditioner)
    if (spla.norm(complex_part)> _SMALL_NUMBER):
      print("Warning. There may be numerical issues.  Preconditioner has complex values")
      print("Norm of the imaginary component is, ")+str(spla.norm(complex_part))
    preconditioner_sqrt=np.real(preconditioner_sqrt)
      
    for i in range(self.step_count):
      g = -self._compute_stochastic_gradient(self.x) #,a)
      scaled_grad=preconditioner.dot(g)
      scaled_noise = preconditioner_sqrt.dot(np.random.randn(self.dim)) 
      self.x = self.x + self.step_size * scaled_grad+np.sqrt(2*self.step_size)*scaled_noise
      sys.stdout.write('*')
    return self.x

  def update_observation(self, context, article, feedback):
    '''updates the observations for displayed article, given the context and 
    user's feedback. The new observations are saved in the history of the 
    displayed article and the current map estimate and Hessian of this 
    article are updated right away.
    
    Args:
      context - a list containing observed context vector for each article
      article - article which was recently shown
      feedback - user's response.
      '''
    if self.time:
      start = time.time()
    self.num_plays +=1
      #[article] += 1
    self.contexts = np.append(self.contexts,[context[article]], axis=0)
    self.rewards = np.append(self.rewards, [feedback])
    z = context[article]
    pred = 1/(1+np.exp(-self.x.dot(z)))
    g = (pred - feedback) * z
    #print(np.shape(self.gradient), np.shape(g))
    self.gradients = np.append(self.gradients, [g], axis=0)
    self.gradient += g
    
  def _compute_stochastic_gradient(self, x):
    '''computes a stochastic gradient of the negative log-posterior for the given
     article.'''
    if self.num_plays<=self.batch_size:
      sample_indices = range(self.num_plays)
      gradient_scale = 1
    else:
      gradient_scale = self.num_plays/self.batch_size
      sample_indices = rnd.sample(range(self.num_plays),self.batch_size)
    
    old_gradients = self.gradients[sample_indices]
    
    zs = self.contexts[sample_indices].T
    ys = self.rewards[sample_indices]
    preds = 1/(1+np.exp(-x.dot(zs)))
    if self.num_plays <= self.batch_size:
        grads = np.diag(preds - ys).dot(zs.T)
        self.gradients = grads
        g = np.sum(grads, axis = 0)
        self.gradient = g
        #g = zs.dot(preds - pred_xstar)
    else:
        grads = np.diag(preds - ys).dot(zs.T)
        old_grad_sum = np.sum(self.gradients[sample_indices], axis=0)
        self.gradients[sample_indices] = grads
        new_grad_sum = np.sum(grads, axis = 0)
        g = self.gradient + gradient_scale * (new_grad_sum - old_grad_sum) #variance-reduced gradient
        self.gradient = self.gradient + (new_grad_sum - old_grad_sum)
                      
        #variance-reduced gradient
#     g = np.zeros(self.dim)
#     for i in sample_indices: #should parallelize this!
#       z = self.contexts[i]
#       y = self.rewards[i]
#       pred = 1/(1+np.exp(-x.dot(z)))
#       if self.num_plays[article]<=self.batch_size:
#         g = g + (pred-y)*z
#       else:
#         pred_xstar = 1/(1+np.exp(-xstar.dot(z)))
#         g = g + (pred - pred_xstar)*z #variance-reduced gradient
    #TODO: don't calculate hessian every step.
    g_prior,_ = self._compute_gradient_hessian_prior(x)
    g = g + g_prior
    return g  

##################################### 12/25
class DefaultAgent(Agent):
    def __init__(self,dim):
        self.dim=dim
        self.num_plays = 0
        self.contexts = np.zeros((0,self.dim))
        self.rewards = np.asarray([]) 
    def update_observation(self, context, article, feedback):
        self.num_plays +=1
        self.contexts = np.append(self.contexts,[context[article]], axis=0)
        self.rewards = np.append(self.rewards, [feedback])
    def pick_action(self,context):
        return 0

class ThompsonSampler(Agent):
  """Greedy News Recommender."""
  
  def __init__(self,num_articles,dim):
    self.num_articles = num_articles
    self.dim = dim
  
    # keeping the observations for each article
    self.num_plays = 0
    self.contexts = np.zeros((0,self.dim))
    self.rewards = np.asarray([]) 
    
  def get_sample(self):
    return
    #subclass should define this
  
  def update_observation(self, context, article, feedback):
    '''updates the observations for displayed article, given the context and 
    user's feedback. The new observations are saved in the history of the 
    displayed article.
    
    Args:
      context - a list containing observed context vector for each article
      article - article which was recently shown
      feedback - user's response.
      '''
    self.num_plays +=1
    self.contexts = np.append(self.contexts,[context[article]], axis=0)
    self.rewards = np.append(self.rewards, [feedback])
  
  def pick_action(self,context):
    '''Greedy action based on sample.'''
    sample = self.get_sample()
    sample_rewards = [evaluate_log1pexp(np.dot(sample, ctxt)) for ctxt in context]
    article = np.argmax(sample_rewards)
    #if self.time:
    #  end=time.time()
    #  print("(pick_action, %s) Time Elapsed: %f" % (end - start))
    return article


class BasicLangevinTS(ThompsonSampler):
    def __init__(self, num_articles, dim, mu, cov=None, step_size=0.1, n_steps=100, init_pt=None):
        ThompsonSampler.__init__(self, num_articles, dim)
        self.mu = mu
        self.cov = cov
        self.step_size = step_size
        self.n_steps = n_steps
        self.theta = init_pt
        
    def get_sample(self):
        self.theta = langevin(self.dim, [self.contexts, self.rewards], logistic_grad_f, Gaussian_prior_grad_f(self.mu), 
                              step_size = self.step_size / (self.num_plays + 1), n_steps = self.n_steps, init_pt = self.theta)
        return self.theta
    










