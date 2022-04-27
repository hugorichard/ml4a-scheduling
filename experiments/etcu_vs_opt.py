"""Experiments."""

# TODO 1, 2, FTP correct, FTP wrong
import numpy as np
from mlforscheduling.etc_u import opt, etc_u, flow_time
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 2000
lambda1 = np.linspace(0.01, 1, 100)
lambda2 = 1

all_flow_times = []
for seed in tqdm(range(100)):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [etc_u]:
        flow_times_alg = []
        for sc in lambda1:
            jobs1 = rng.exponential(scale=sc, size=n)
            jobs2 = rng.exponential(scale=1, size=n)
            flow_times_alg.append(flow_time(algo(jobs1, jobs2)) / flow_time(opt(jobs1, jobs2)))
        flow_times.append(flow_times_alg)
    all_flow_times.append(flow_times)

all_flow_times = np.array(all_flow_times)

flow_times_high = np.quantile(all_flow_times, 0.9, axis=0)
flow_times_low = np.quantile(all_flow_times, 0.1, axis=0)
flow_times = np.median(all_flow_times, axis=0)

plt.figure()
plt.plot(lambda1, flow_times[0], label="ETC-U")
plt.fill_between(lambda1, flow_times_low[0],flow_times_high[0], alpha=0.3)
plt.axvline(1/(1 + 2*np.sqrt(2)))
plt.axhline(2)
plt.xlabel("Mean processing time of job 1 (the other job has mean 1)")
plt.ylabel("CR")
plt.legend()
plt.show()


