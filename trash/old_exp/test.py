"""Reproducing Figure 1: run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.ucb_rr import ucb_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr

n = 1000
lambda1 = 0.0001
lambda2 = 1
rng = np.random.RandomState(2)
jobs1 = rng.exponential(scale=lambda1, size=n)
jobs2 = rng.exponential(scale=1, size=n)
flowtime_rr =[]
flowtime_ftpp =[]
flowtime_ftpp_test =[]

def custom_ftpp(jobs):
    flow = 0 
    flow+= (np.flip(np.arange(n))+1+n)@jobs[0,:]+(np.flip(np.arange(n))+1)@jobs[1,:]
    return flow

for seed in range(1000,1100):
    rng = np.random.RandomState(seed)
    jobs1 = rng.exponential(scale=lambda1, size=n)
    jobs2 = rng.exponential(scale=1, size=n)
    flowtime_rr.append(rr(np.vstack([jobs1,jobs2])))
    flowtime_ftpp.append(ftpp(np.vstack([jobs1,jobs2])))
    flowtime_ftpp_test.append(custom_ftpp(np.vstack([jobs1,jobs2])))

print("ftpp", np.mean(flowtime_ftpp))
print("ftpp test", np.mean(flowtime_ftpp_test))

print("rr", np.mean(flowtime_rr))
print("diff",np.mean(flowtime_rr)-np.mean(flowtime_ftpp))
