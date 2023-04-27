"""Reproducing Figure 1: run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.ucb_rr import ucb_rr
from mlforscheduling.etc_u import etc_u


from mlforscheduling.utils import ftpp, opt, rr

n = 100
lambda1 = np.linspace(0.02, 0.4, 10)
lambda2 = 1
seeds = np.arange(300)



def do_stuff(seed, lambda1, lambda2, n):
    flow_times = []
    for algo in [etc_u, etc_rr, ucb_u, ucb_rr,rr, ftpp]:
        flow_times_alg = []
        for sc in lambda1:
            rng = np.random.RandomState(seed)
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            flow_times_alg.append(
                algo(np.vstack([jobs1,jobs2])) / opt(np.vstack([jobs1, jobs2]))
            )
        flow_times.append(flow_times_alg)
    return flow_times

all_flow_times = Parallel(n_jobs=10, verbose=True)(
   delayed(do_stuff)(seed, lambda1, lambda2, n) for seed in seeds
)
all_flow_times = np.array(all_flow_times)
np.save("./data/etc_vs_opt-2-jobs.npy", all_flow_times)
