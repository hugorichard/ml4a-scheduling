from numba import jit
import numpy as np 



@jit(nopython=True)
def klBern_numba(x, y):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


@jit
def klucb_numba(x, d, kl, upperbound=1,
                lowerbound=0, precision=1e-4, max_iterations=50):
#lowerbound version
    value = min(x, upperbound)
    u = lowerbound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) / 2.
        if kl(x, m) < d:
            u = m
        else:
            value = m
    return (value + u) / 2.



@jit
def klucbBern_numba(x, d, precision=1e-4):
    upperbound = min(1., klucbGauss_numba(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    lowerbound = 0
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb_numba(x, d, klBern_numba, upperbound, precision)

