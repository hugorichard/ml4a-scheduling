"""Reproducing Figure 1: run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr
from mlforscheduling.lsept import lsept
from mlforscheduling.lsept import lsept
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.ucb_rr import ucb_rr
n = 1000
lambda1 = np.linspace(0.1, 1, 10)
lambda2 = 1
seeds = np.arange(200)
def do_stuff(seed, lambda1, lambda2, n):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [ftpp, etc_u, etc_rr, rr,  lsept,ucb_u,ucb_rr, opt]:#
        flow_times_alg = []
        for sc in lambda1:
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            flow_times_alg.append(
                algo(np.vstack([jobs1, jobs2]))
            )
        flow_times.append(flow_times_alg)
    return flow_times
all_flow_times = Parallel(n_jobs=10, verbose=True)(
    delayed(do_stuff)(seed, lambda1, lambda2, n) for seed in seeds
)
all_flow_times = np.array(all_flow_times)
np.save("./data/etc_vs_opt-2-jobs-rebuttal.npy", all_flow_times)
