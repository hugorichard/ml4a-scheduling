"""Reproducing Figure 2 (right): run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.ucb_rr import ucb_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr


ns = np.linspace(start=2, stop=10, num=10)
n=300
k = 2
seeds = np.arange(1000)
M = [1]


def do_stuff(seed, n, k, M):
    flow_times = []
    for algo in [etc_u, etc_rr, ucb_u,ucb_rr,rr, ftpp,opt]:
        rng = np.random.RandomState(seed)
        jobs = []
        for i in range(k):
            jobs.append(rng.exponential(scale=(i+1)/k, size=n))
        jobs = np.array(jobs)
        flow_times.append(algo(jobs))
    return flow_times


results = Parallel(n_jobs=10, verbose=True)(
    delayed(do_stuff)(seed, int(n), int(k), M) for seed in seeds for k in ns
)
results = np.array(results).reshape(len(seeds), len(ns), 7)
np.save("./data/etc_vs_opt-vary-ktypes-unnormalized.npy", results)
