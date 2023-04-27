"""Reproducing Figure 2 (right): run experiment."""

import numpy as np
from joblib import Parallel, delayed
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.ucb_rr import ucb_rr
from mlforscheduling.etc_u import etc_u
from mlforscheduling.utils import ftpp, opt, rr


ns = np.logspace(start=1, stop=4, num=10)
print(ns)
k = 2
seeds = np.arange(400)


def do_stuff(seed, n, k):
    flow_times = []
    print(n)
    for algo in [etc_u, etc_rr, ucb_u, ucb_rr,rr, ftpp,opt]:
        rng = np.random.RandomState(seed)
        lambdas = [0.25,1]
        jobs = []
        for i in range(k):
            jobs.append(rng.exponential(scale=lambdas[i], size=n))
        jobs = np.array(jobs)
        flow_times.append(algo(jobs))
    return flow_times


results = Parallel(n_jobs=10, verbose=True)(
    delayed(do_stuff)(seed, int(n), k) for seed in seeds for n in ns
)
results = np.array(results).reshape(len(seeds), len(ns), 7)
np.save("./data/all_algos_vary_nsamples.npy", results)
