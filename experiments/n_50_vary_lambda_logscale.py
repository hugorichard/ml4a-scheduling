"""Reproducing Figure 1: run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.ucb_rr import ucb_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr

n = 50
lambda1 = [0.01,0.05,0.1,0.3,0.6,0.8,0.9,0.95,1.]
lambda2 =1
seeds = np.arange(50000)


def do_stuff(seed, lambda1, lambda2, n):
    algos =[etc_u, etc_rr, ucb_u, ucb_rr,rr, ftpp,opt]
    flow_times = []
    for i in range(7):
        flow_times_alg = []
        algo = algos[i]
        for sc in lambda1:
            rng = np.random.RandomState(seed)
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            if i==3:
                flow_times_alg.append( algo(np.vstack([jobs1,jobs2]),sc/20))
            else:
                flow_times_alg.append(
                    algo(np.vstack([jobs1,jobs2])))
        flow_times.append(flow_times_alg)
    return flow_times

all_flow_times = Parallel(n_jobs=10, verbose=True)(
   delayed(do_stuff)(seed, lambda1, lambda2, n) for seed in seeds
)
all_flow_times = np.array(all_flow_times)
np.save("./data/n_50_vary_lambdas_logscale.npy", all_flow_times)
