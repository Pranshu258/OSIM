# Implement the Oblivious Adversary here
# For Lipschitz Function we are using the 

import numpy as np
from scipy.stats import logistic


class Adversary:
    dimensions = 5
    data_size = 10000
    model = []
    train_data = []
    test_data = []

    def get_rand_vec(self, dims):
        x = np.random.standard_normal(dims)
        r = np.sqrt((x*x).sum())
        return x / r

    def gen_data(self, size):
        data = []
        for i in range(0, self.data_size):
            x = np.random.standard_normal(self.dimensions)
            r = np.sqrt((x*x).sum())
            w = np.random.randint(1, high=self.W)
            x = w * x / r
            z = np.dot(self.model, x)
            y = 4*logistic.cdf(z)
            data.insert(len(data), [x, z, y])
        return data

    def __init__(self, d, N, W):
        self.dimensions = d
        self.data_size = N
        self.model = self.get_rand_vec(d)
        self.W = W
        self.train_data = self.gen_data(N)
        self.test_data = self.gen_data(N/10)
        

adv = Adversary(5, 1000, 5)

# for i in range(0, adv.data_size):
#     print(adv.train_data[i][0], "\t", adv.train_data[i][1], "\t", adv.train_data[i][2])