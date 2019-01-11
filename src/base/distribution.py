
import numpy as np
#import scipy as sp

class Distribution(object):
    def __init__(self):
        pass
    def __call__(self):
        pass
        #return self.generate()
    #def generate(self):
    #    pass

class NormalDist(Distribution):
    def __init__(self, mean, var, dim=None):
        self.mean=mean
        self.var=var
        self.dim=dim
    def __call__(self): 
        #generate(self):
        if self.dim==None:
            return np.random.multivariate_normal(self.mean,self.var)
        else:
            #return np.random.multivariate_normal(np.full(dim,mean),np.diag(np.full(dim,mean)))
            return np.random.normal(loc=self.mean, scale=np.sqrt(self.var), size=self.dim)

class BernoulliDist(Distribution):
    def __init__(self, p, dim):
        self.p=p # self.dim-1
        self.dim=dim
    def __call__(self):
        return np.random.binomial(1,self.p,self.dim)

class DistributionWithConstant(Distribution):
    def __init__(self,dist,const=1):
        self.dist = dist
        self.const = const
    def __call__(self):
        return np.append([self.const], self.dist())
