"""Experiments."""

# TODO 1, 2, FTP correct, FTP wrong
import matplotlib.pyplot as plt
import numpy as np
from mlforscheduling.etc_u import etc_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.utils import opt, ftpp
from tqdm import tqdm

n = 1000
k = 4
concentrations = np.linspace(2/20, 2, 20) # Dirichelet parameter

all_flow_times = []
for seed in tqdm(range(20)):
    rng = np.random.RandomState(seed)
    flow_times = []
    for algo in [etc_u, etc_rr, ftpp]:
        flow_times_alg = []
        for concentration in concentrations:
            lambdas = rng.dirichlet(np.ones(k) * concentration)
            jobs = []
            for i in range(k):
                jobs.append(rng.exponential(scale=lambdas[i], size=n))
            jobs = np.array(jobs)
            flow_times_alg.append(
                algo(jobs)
                / opt(jobs)
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
    plt.plot(concentrations, flow_times[i], label=algo)
    plt.fill_between(concentrations, flow_times_low[i], flow_times_high[i], alpha=0.3)
    plt.xlabel("Concentration of processing time means")
    plt.ylabel("CR")
plt.legend()
plt.show()
