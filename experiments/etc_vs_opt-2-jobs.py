"""Experiments."""

# TODO 1, 2, FTP correct, FTP wrong
import matplotlib.pyplot as plt
import numpy as np
from mlforscheduling.etc_u import etc_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.utils import opt, ftpp, rr
from joblib import Parallel, delayed

n = 2000
lambda1 = np.linspace(0.02, 1, 20)
lambda2 = 1
seeds = np.arange(100)


def do_stuff(seed, lambda1, lambda2, n):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [etc_u, etc_rr, rr, ftpp]:
        flow_times_alg = []
        for sc in lambda1:
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            flow_times_alg.append(
                algo(np.vstack([jobs1, jobs2])) / opt(np.vstack([jobs1, jobs2]))
            )
        flow_times.append(flow_times_alg)
    return flow_times


all_flow_times = Parallel(n_jobs=10, verbose=True)(
    delayed(do_stuff)(seed, lambda1, lambda2, n) for seed in seeds
)
all_flow_times = np.array(all_flow_times)
np.save("./data/etc_vs_opt-2-jobs.npy", all_flow_times)
