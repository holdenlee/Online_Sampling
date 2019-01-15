import numpy.linalg as npla
import numpy as np
import time

for i in range(100):
    start = time.time()
    npla.inv(np.random.random((20,20)))
    #print('hi', 999+999)
    end = time.time()
    print(end-start)