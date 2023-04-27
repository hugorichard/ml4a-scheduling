"""Reproducing Figure 2 (right): run experiment."""

import numpy as np
from joblib import Parallel, delayed

from mlforscheduling.lsept import lsept

from mlforscheduling.utils import ftpp, opt, rr


ns = np.logspace(start=2, stop=6, num=10)
k = 2
seeds = np.arange(200)
M = [1]


def do_stuff(seed, n, k, M):
    flow_times = []
    for algo in [rr, ftpp,lsept]:
        rng = np.random.RandomState(seed)
        lambdas = [0.8,1]
        jobs = []
        for i in range(k):
            jobs.append(rng.exponential(scale=lambdas[i], size=n))
        jobs = np.array(jobs)
        flow_times.append(algo(jobs) / opt(jobs))
    return flow_times


results = Parallel(n_jobs=10, verbose=True)(
    delayed(do_stuff)(seed, int(n), k, M) for seed in seeds for n in ns
)
results = np.array(results).reshape(len(seeds), len(ns), 3)
np.save("./data/vary-nsamples_lsept.npy", results)
