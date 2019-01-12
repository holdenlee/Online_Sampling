from random import *
import math
from utils import *

def get_level(n):
    return math.ceil(math.log(n+1,2))

def is_power_2(n):
    return (n==0 or 2**math.floor(math.log(n,2))==n)

class RandomWeights(object):
    def __init__(self):
        self.arr = []
        self.levels=-1
        self.next=0
    def add(self,wt):
        if is_power_2(self.next):
            self.arr += [[0]]
            self.levels +=1
        index = self.next
        for l in range(self.levels+1):
            if len(self.arr[l]<=index):
                self.arr[l]+=[0]
            self.arr[l][index]+=wt
            index = index / 2
        self.next += 1
    def adjust(self,n,wt):
        diff = wt - self.arr[0][n]
        index = n
        for l in range(self.levels+1):
            self.arr[l][index]+=diff
            index = index / 2
    def sample(self):
        r = random.random() * self.arr[self.levels][0]
        i=0
        for l in range(self.levels-1, 0,-1):
            cutoff = self.arr[l][2*i]
            if r < cutoff:
                i = 2*i
            else:
                r -= cutoff
                i = 2*i+1
        return i
    def __repr__(self):
        return unlines([repr(s) for s in arr])
    
