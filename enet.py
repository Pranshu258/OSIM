# The sequential sum of distance between a point and the point closest to it in a ball is bounded 
# Points are received stochastically one by one

# Author: Pranshu Gupta

import numpy as np
import math
from scipy.spatial.distance import cdist

def get_rand_vec(dims, R):
    x = np.random.standard_normal(dims)
    m = np.sqrt((x*x).sum()) # norm of the vector x
    r = R*np.random.random() # upper bound of the vector ball
    x = r * x / m 
    return x

def find_closest_distance(x, X):
    return pow(cdist([x], X).min(), 2)

d = 3
R = 1
X = [get_rand_vec(d, R)]
S = 0.0
for t in range (100000):
    x = get_rand_vec(d, R)
    dist = find_closest_distance(x, X)
    X.insert(len(X), x)
    S = S + dist
    if t%10 == 0:
        print(t, S)
        