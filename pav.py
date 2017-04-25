import numpy as np
import bisect
import math
from adversary import Adversary

# Author : Alexandre Gramfort
# license : BSD


def pav(y):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y
    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v


if __name__ == '__main__':
    T = 200
    D = 5
    R = 5
    adv = Adversary(D, T, R)
    E = 0
    for t in range(1,T):
        X, Z, Y = adv.train_data[0][:t], adv.train_data[1][:t], adv.train_data[2][:t]
        sZ, sY = zip(*sorted(zip(Z,Y)))
        sZ, sX = zip(*sorted(zip(Z,X)))
        # fit the PAVA
        pavfit = pav(sY).tolist()
        # get the error of t'th point
        x, z, y = adv.train_data[0][t], adv.train_data[1][t], adv.train_data[2][t]
        # search z in sZ and get the adjacent point
        pos = bisect.bisect_left(sZ, z)
        if pos == 0:
            err = math.pow(pavfit[0] - y, 2)
            E = E + err
            print(t, err, E)
        elif pos == len(sZ):
            err = math.pow(sY[pavfit(sY)-1] - y, 2)
            E = E + err
            print(t, err, E)
        else:
            z1, z2, y1, y2 = sZ[pos-1], sZ[pos], pavfit[pos-1], pavfit[pos]
            err = math.pow(y1 + (y2-y1)*(z-z1)/(z2-z1) - y, 2)
            E = E + err
            print(t, err, E)
    

    # import pylab as pl
    # pl.close('all')
    # pl.plot(sZ, sY, 'rx')
    # pl.plot(sZ, pavfit, 'b')
    # pl.show()