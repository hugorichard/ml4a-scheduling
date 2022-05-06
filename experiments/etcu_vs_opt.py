"""Experiments."""

# TODO 1, 2, FTP correct, FTP wrong
import matplotlib.pyplot as plt
import numpy as np
from mlforscheduling.etc_u import etc_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.utils import opt, ftpp
from tqdm import tqdm

n = 1000
lambda1 = np.linspace(0.01, 1, 20)
lambda2 = 1

all_flow_times = []
for seed in tqdm(range(20)):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [etc_u, etc_rr, ftpp]:
        flow_times_alg = []
        for sc in lambda1:
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            flow_times_alg.append(
                algo(lambda n: 6 * n**2, np.vstack([jobs1, jobs2]))
                / opt(np.vstack([jobs1, jobs2]))
            )
        flow_times.append(flow_times_alg)
    all_flow_times.append(flow_times)

# %%

all_flow_times = np.array(all_flow_times)

# %%

print(all_flow_times.shape)

# %%

flow_times_high = np.quantile(all_flow_times, 0.9, axis=0)
flow_times_low = np.quantile(all_flow_times, 0.1, axis=0)
flow_times = np.median(all_flow_times, axis=0)

# %%

plt.figure()
for i, algo in enumerate(["ETC-U", "ETC-RR", "FTPP"]):
    plt.plot(lambda1, flow_times[i], label=algo)
    plt.fill_between(lambda1, flow_times_low[i], flow_times_high[i], alpha=0.3)
    plt.axvline(1 / (1 + 2 * np.sqrt(2)))
    plt.axhline(2)
    plt.xlabel("Mean processing time of job 1 (the other job has mean 1)")
    plt.ylabel("CR")
plt.legend()
plt.show()
