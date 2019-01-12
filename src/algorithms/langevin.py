import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
import random as rnd
import time

#https://github.com/b2du/LangevinTS/blob/master/code/SAGALD.py

_LARGE_NUMBER = 1e+2

def evaluate_log1pexp(x):
    """given the input x, returns log(1+exp(x))."""
    return np.piecewise(x, [x>_LARGE_NUMBER], [lambda x: x, lambda x: np.log(1+np.exp(x))])
    """
    if x > _LARGE_NUMBER:
      return x
    else:
      return np.log(1+np.exp(x))
    """
    
def evaluate_logistic(m):
    return np.piecewise(m, [m>_LARGE_NUMBER], [lambda x: 0, lambda x: 1/(1+np.exp(-x))])

# x : R^d
# data = (contexts : (R^d)^T, rewards : (R^d)^T)
def logistic_grad_f(x, data):
    zs = data[0]
    ys = data[1]
    m = x.dot(zs.T)
    preds = evaluate_logistic(m)
    #preds = 1/(1+np.exp(-x.dot(zs.T)))
    #print(np.shape(preds), np.shape(ys), np.shape(zs))
    grads = np.diag(preds - ys).dot(zs)
    return grads

def logistic_f(x, data):
    zs = data[0]
    ys = data[1]
    return evaluate_log1pexp(x.dot(zs.T)) - ys * x.dot(zs.T)

class Gaussian_prior_grad_f(object):
    # mu : R^d
    # cov : Maybe (R^d)^d
    def __init__(self, mu, cov=None):
        d = np.shape(mu)[0]
        self.inv_cov = np.eye(d) if cov is None else np.linalg.inv(cov)
        self.mu = mu
    def __call__(self, x):
        #print(np.shape(x), np.shape(self.mu))
        return self.inv_cov.dot(x-self.mu)
    
class Gaussian_prior_f(object):
    # mu : R^d
    # cov : Maybe (R^d)^d
    def __init__(self, mu, cov=None):
        d = np.shape(mu)[0]
        self.inv_cov = np.eye(d) if cov is None else np.linalg.inv(cov)
        self.mu = mu
    def __call__(self, x):
        #print(np.shape(x), np.shape(self.mu))
        return 0.5*(x-self.mu).dot(self.inv_cov.dot(x-self.mu))

def logistic_Hessian(x, z): #data):
    #z = data[0]
    #y = data[1]
    if type(z) == list:
        z = np.asarray(z)
    if len(z.shape)==1:
        return evaluate_log1pexp(x.dot(z)) * evaluate_log1pexp(-x.dot(z)) * np.outer(z,z)
        #z = np.expand_dims(z, 1) #asarray([z]) #z.reshape((#,1))
        #z1 = np.expand_dims(evaluate_log1pexp(x.dot(z)) * evaluate_log1pexp(-x.dot(z)), 1) * z
    else:
        z1 = z.T * evaluate_log1pexp(x.dot(z.T)) * evaluate_log1pexp(-x.dot(z.T))
        return z1.dot(z)
    
def langevin_step(d, data, grad_f, prior_grad_f, 
                  x, step_size = 0.01, Hinv = None):
    grads = grad_f(x, data)
    gradient = np.sum(grads, axis = 0)
    g = gradient
    g_prior = prior_grad_f(x)
    g = g + g_prior
    #preconditioner_sqrt.dot(np.random.randn(self.dim)) 
    if Hinv is None:
        noise = np.random.randn(d)
        x = x - step_size * g + \
            np.sqrt(2*step_size)*noise
    else:
        noise = np.random.multivariate_normal([0]*d,Hinv)
        x = x - step_size * Hinv.dot(g) + \
            np.sqrt(2*step_size)*noise
    return x #, gradients, gradient

def langevin(d, data, grad_f, prior_grad_f, 
             step_size = 0.01, n_steps=100, init_pt=None, time_limit=0.0, H=None):
    if init_pt is None:
        init_pt = np.zeros(d)
    x = init_pt
    start = time.time()
    Hinv = None if H is None else la.inv(H)
    for t in range(n_steps):
        x = langevin_step(d, data, grad_f, prior_grad_f, x, step_size, Hinv)
        if time_limit > 0 and time.time() - start > time_limit:
            break
    return x, t+1

def mala_step(d, data, f, prior_f, grad_f, prior_grad_f, 
                  x, step_size = 0.01):
    grads = grad_f(x, data)
    gradient = np.sum(grads, axis = 0)
    g = gradient
    g_prior = prior_grad_f(x)
    g = g + g_prior
    noise = np.random.randn(d)
    #preconditioner_sqrt.dot(np.random.randn(self.dim)) 
    scaled_noise = np.sqrt(2*step_size)*noise
    z = x - step_size * g + \
        scaled_noise
    grads_z = grad_f(z, data)
    gz = np.sum(grads_z, axis=0) + prior_grad_f(z)
    valx = np.sum(f(x,data), axis=0) + prior_f(x)
    valz = np.sum(f(z,data), axis=0) + prior_f(z)
    p = np.exp(-valz+valx+
               (la.norm(x-z-step_size*gz)**2)/(4*step_size)-la.norm(noise)**2/2)
    (samp, accept) = (z, True) if np.random.random()<p else (x, False)
    #samp = z if np.random.random()<p else x
    return samp, accept, min(p,1)

def mala(d, data, f, prior_f, grad_f, prior_grad_f, 
             step_size = 0.01, n_steps=100, init_pt=None, time_limit=0.0):
    accepts = 0.0
    if init_pt is None:
        init_pt = np.zeros(d)
    x = init_pt
    start = time.time()
    for t in range(n_steps):
        x, accept, _ = mala_step(d, data, f, prior_f, grad_f, prior_grad_f, x, step_size)
        if accept:
            accepts += 1
        if time_limit > 0 and time.time() - start > time_limit:
            break
    return x, accepts/(t+1), t+1

def sgld_step(t, d, data, batch_grad_f, prior_grad_f, 
                  x, step_size = 0.01, batch_size = 32, Hinv = None):
    if t <= batch_size:
      sample_indices = range(t)
      #gradient_scale = 1
    else:
      gradient_scale = t/batch_size
      sample_indices = rnd.sample(range(t),batch_size)
    sampled_data = tuple([arr[sample_indices] for arr in data])
    #g is estimated gradient
    grads = batch_grad_f(x, sampled_data)
    gradient = np.sum(grads, axis = 0)
    g = gradient
    g_prior = prior_grad_f(x)
    g = g + g_prior
    if Hinv is None:
        noise = np.random.randn(d)
        x = x - step_size * g + \
            np.sqrt(2*step_size)*noise
    else:
        noise = np.random.multivariate_normal([0]*d,Hinv)
        x = x - step_size * Hinv.dot(g) + \
            np.sqrt(2*step_size)*noise
    return x #, gradients, gradient

def sgld(t, d, data, grad_f, prior_grad_f, 
             step_size = 0.01, n_steps=100, batch_size = 32, init_pt=None, 
         H=None, time_limit=0.0):
    if init_pt is None:
        init_pt = np.zeros(d)
    x = init_pt
    start = time.time()
    Hinv = None if H is None else la.inv(H)
    for i in range(n_steps):
        x = sgld_step(t, d, data, grad_f, prior_grad_f, x, step_size, batch_size, Hinv)
        if time_limit > 0 and time.time() - start > time_limit:
            break
    return x, i+1


# t : int
# d : int
# gradients : (R^d)^t
# gradient : R^d (sum of gradients)
# data : a (tuple of numpy arrays) 
    # todo - also generalize to numpy array - for example, (R^(d'))^t - or 
# batch_grad_f : (R^d) -> a -> (R^d)^(batch_size)
# prior_grad_f : R^d -> R^d
# x : R^d
# batch_size : int
# step_size : float
def sagald_step(t, d, gradients, gradient, data, batch_grad_f, prior_grad_f, 
                x, batch_size = 32, step_size = 0.01, Hinv=None):
    #print("1: ",t,batch_size)#debug
    if t <= batch_size:
      sample_indices = range(t)
      #gradient_scale = 1
    else:
      gradient_scale = t/batch_size
      sample_indices = rnd.sample(range(t),batch_size)
    #print(sample_indices)#debug
    
    old_gradients = gradients[sample_indices]
    sampled_data = tuple([arr[sample_indices] for arr in data])
    # ex. this is (zs, ys)
    #zs = self.contexts[sample_indices] # .T
    #ys = self.rewards[sample_indices]
    #to generalize to tuple of arrays OR array,
    #if isinstance(data, np.ndarray)
    
    #g is estimated gradient
    grads = batch_grad_f(x, sampled_data)
    if t <= batch_size:
        gradients[sample_indices] = grads
        gradient = np.sum(grads, axis = 0)
        g = gradient
    else:
        old_grad_sum = np.sum(gradients[sample_indices], axis=0)
        gradients[sample_indices] = grads #this mutates gradients!
        new_grad_sum = np.sum(grads, axis = 0)
        g = gradient + gradient_scale * (new_grad_sum - old_grad_sum) #variance-reduced gradient
        gradient = gradient + (new_grad_sum - old_grad_sum) 
            #warning, this doesn't mutate, need to do it in the calling method
    g_prior = prior_grad_f(x)
    #print(np.shape(g),np.shape(g_prior),'grad_shape')
    g = g + g_prior
    if Hinv is None:
        noise = np.random.randn(d)
        x = x - step_size * g + \
            np.sqrt(2*step_size)*noise
    else:
        #Hinv = la.inv(H)
        noise = np.random.multivariate_normal([0]*d,Hinv)
        x = x - step_size * Hinv.dot(g) + \
            np.sqrt(2*step_size)*noise
    #print(x)
    return x, gradients, gradient

# t : int
# d : int
# data : a (tuple of numpy arrays) 
    # todo - also generalize to numpy array - for example, (R^(d'))^t - or 
# gradients : (R^d)^t
# gradient : R^d (sum of gradients)
# batch_grad_f : (R^d) -> a -> (R^d)^(batch_size)
# prior_grad_f : R^d -> R^d
# init_pt : R^d
# batch_size : int
# step_size : float
# num_steps : int
# max_time : float (in seconds)
def sagald(t, d, data, batch_grad_f, prior_grad_f, 
           gradients = None, gradient = None,
           batch_size = 32, step_size = 0.01, n_steps = 200,
           init_pt = None,
           H = None,
           time_limit = 0.0):
    x = init_pt
    if x is None:
        x = np.zeros(d)
    if gradients is None:
        gradients = batch_grad_f(x, sampled_data)
    if gradient is None:
        gradient = np.sum(gradients, axis=0)
    start = time.time()
    Hinv = None if H is None else la.inv(H) 
    for i in range(n_steps):
        (x, gradients, gradient) = \
             sagald_step(t, d, gradients, gradient, data, 
                         batch_grad_f, prior_grad_f, 
                         x, batch_size = batch_size, step_size = step_size, Hinv=Hinv)
        if time_limit > 0 and time.time() - start > time_limit:
            break
    return x, gradients, gradient, i+1