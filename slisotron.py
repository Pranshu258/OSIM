# Implement the Batch Learner here
from pav import pav
import numpy as np
import bisect
import math
from adversary import Adversary


def slisotron(w, X, Y):
    sZ = []
    PAVY = []
    for t in range(100):
        # get the z as per the current weight model
        Z = [np.dot(w, x) for x in X]
        # sort these points as per z values
        sZ, sY = zip(*sorted(zip(Z,Y)))
        # Z = np.array(Z)
        # X = np.array(X)
        # print (Z.shape)
        # print (X.shape)
        # print (len(X))
        # print (list(zip(Z,list(X))))
        a = list(zip(Z,X))
        # print (a[0][0])
        # print (a[0][1])
        sZ, sX = zip(*sorted(a, key=lambda x: x[0]))
        # apply PAV
        PAVY = pav(sY)
        for i in range(len(sX)):
            w = w + (sY[i] - PAVY[i])*sX[i]/len(sX)
    return [w, sZ, PAVY]


T = 200
D = 5
R = 5
adv = Adversary(D, T, R)
E = 0.0
w = np.array([0 for x in range(D)])
for t in range(1,T):
    X, Z, Y = adv.train_data[0][:t], adv.train_data[1][:t], adv.train_data[2][:t]
    w, sZ, PAVY = slisotron(w, X, Y)
    # get the error of t'th point
    x, z, y = adv.train_data[0][t], adv.train_data[1][t], adv.train_data[2][t]
    # search z in sZ and get the adjacent point
    pos = bisect.bisect_left(sZ, z)
    if pos == 0:
        err = math.pow(PAVY[0] - y, 2)
        E = E + err
        print(t, err, E)
    elif pos == len(sZ):
        err = math.pow(PAVY[len(sZ)-1] - y, 2)
        E = E + err
        print(t, err, E)
    else:
        z1, z2, y1, y2 = sZ[pos-1], sZ[pos], PAVY[pos-1], PAVY[pos]
        err = math.pow(y1 + (y2-y1)*(z-z1)/(z2-z1) - y, 2)
        E = E + err
        print(t, err, E)


