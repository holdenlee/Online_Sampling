from random import *
import math
from utils import *
import copy

def get_level(n):
    return math.ceil(math.log(n+1,2))

def is_power_2(n):
    return (2**math.floor(math.log(n,2))==n)

class RandomWeights(object):
    #Note everything is 0-indexed.
    def __init__(self):
        self.arr = []
        self.levels=-1
        self.next=0
    def add(self,wt):
        if self.levels==-1:
            self.arr = [[wt]]
            self.levels = 0
            self.next=1
            return
        if is_power_2(self.next):
            self.arr += [list(self.arr[self.levels])] #NEED TO COPY!
            self.levels +=1
        index = self.next
        for l in range(self.levels+1):
            if len(self.arr[l])<=index:
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
        r = random() * self.arr[self.levels][0]
        i=0
        for l in range(self.levels-1,-1,-1):
            cutoff = self.arr[l][2*i]
            if r < cutoff:
                i = 2*i
            else:
                r -= cutoff
                i = 2*i+1
        return i
    def sample_wo_replacement(self, n):
        if n>self.next:
            raise Exception('Not enough elements')
        if n==self.next:
            return list(range(n))
        temp = copy.deepcopy(self.arr) #copy the nested list
        samples = []
        for i in range(n):
            s = self.sample()
            samples += [s]
            self.adjust(s,0)
        self.arr=temp
        return samples
    def sample_w_replacement(self,n):
        return [self.sample() for _ in range(n)]
    def __repr__(self):
        return unlines([repr(s) for s in self.arr])
    def __get__(self,i):
        return self.arr[i]/self.arr[self.levels][0]
    
def test_random_weights():
    rw = RandomWeights()
    rw.add(0.5)
    print(rw)
    rw.add(1)
    print(rw)
    rw.add(2)
    print(rw)
    rw.add(1)
    print(rw)
    rw.adjust(1,1.5)
    print(rw)
    rw.add(1.2)
    print(rw)
    rw.adjust(3,3)
    print(rw)
    print([rw.sample() for _ in range(100)])
    print([rw.sample_wo_replacement(3) for _ in range(100)])

if __name__=='__main__':
    test_random_weights()