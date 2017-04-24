# Implement the Oblivious Adversary here
# For Lipschitz Function we are using the 

import numpy as np
from scipy.stats import logistic


class Adversary:
    dimensions = 5
    data_size = 10000
    model = []              # model is the w in y = u(<w,x>)
    train_data = []
    test_data = []

    def get_rand_vec(self, dims):
        x = np.random.standard_normal(dims)
        r = np.sqrt((x*x).sum())
        return x / r

    def gen_data(self, size):
        X, Z, Y = [], [], []
        for i in range(0, self.data_size):
            x = np.random.standard_normal(self.dimensions)
            m = np.sqrt((x*x).sum()) # norm of the vector x
            r = np.random.randint(1, high=self.R) # upper bound of the vector ball
            x = r * x / m               # a random vector in the ball
            z = np.dot(self.model, x)    # the z value for this vector
            y = 4*logistic.cdf(z)           # the y value for this vector
            X.insert(len(X), x)
            Y.insert(len(Y), y)
            Z.insert(len(Z), z)
        data = [X, Z, Y]
        return data

    def __init__(self, d, N, R):
        self.dimensions = d
        self.data_size = N
        self.model = self.get_rand_vec(d)
        self.R = R
        self.train_data = self.gen_data(N)
        self.test_data = self.gen_data(N/10)
        

# adv = Adversary(5, 1000, 5)

# for i in range(0, adv.data_size):
#     print(adv.train_data[i][0], "\t", adv.train_data[i][1], "\t", adv.train_data[i][2])