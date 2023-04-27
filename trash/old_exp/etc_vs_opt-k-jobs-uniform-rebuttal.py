"""Reproducing Figure 2 (left): run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr
from mlforscheduling.lsept import lsept

n = 2000
ks = np.arange(2, 11)
seeds = np.arange(100)
M = [1]


def do_stuff(seed, n, k, M):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [etc_u, etc_rr, ftpp, rr, lsept]:
        lambdas = np.random.rand(k) * M
        jobs = []
        for i in range(k):
            jobs.append(rng.exponential(scale=lambdas[i], size=n))
        jobs = np.array(jobs)
        flow_times.append(algo(jobs) / opt(jobs))
    return flow_times


results = Parallel(n_jobs=12, verbose=True)(
    delayed(do_stuff)(seed, n, k, M) for seed in seeds for k in ks
)

results = np.array(results).reshape(len(seeds), len(ks), 5)
np.save("./data/etc_vs_opt-k-jobs-uniform-rebuttal.npy", results)
