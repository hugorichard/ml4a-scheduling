import matplotlib.pyplot as plt
import numpy as np
from plotting.params import COLORS

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True,
}
plt.rcParams.update(rc)

ns = np.logspace(start=1, stop=4, num=10)

results = np.load("../experiments/data/all_algos_vary_nsamples.npy")
all_flow_times = np.array(results)
n_seeds = len(all_flow_times)
ns,n_lambda,_ = np.shape(all_flow_times)
flow_times = np.mean(all_flow_times, axis=0)
flow_times_low = np.zeros_like(flow_times)
flow_times_high = np.zeros_like(flow_times)
deviation = np.std(all_flow_times, axis=0)/np.sqrt(ns)

for i in range(6):
    for j in range(n_lambda):
        flow_times[j,i] = flow_times[j,i]/flow_times[j,-1]
        flow_times_low[j,i] = flow_times[j,i] -2*deviation[j,i]/flow_times[j,-1]
        flow_times_high[j,i] = flow_times[j,i] +2*deviation[j,i]/flow_times[j,-1]



indices = {}
for i, algo in enumerate(["ETC-U", "ETC-RR", "UCB-U","UCB-RR", "RR","FTPP"]):
    indices[algo] = i

lambda1 = [   10.,            21.5443469 ,    46.41588834,   100.,
   215.443469,     464.15888336,  1000.,          2154.43469003,
  4641.58883361, 10000.        ]

plt.figure()
for ii, algo in enumerate(["ETC-U", "ETC-RR", "UCB-U","UCB-RR", "RR","FTPP"]):
    i = indices[algo]
    plt.semilogx(lambda1, flow_times[:, i], label=algo, color=COLORS[algo])
    plt.fill_between(lambda1, flow_times_low[:,i], flow_times_high[:,i], alpha=0.3, color=COLORS[algo])
    plt.xlabel("Number of jobs per type ($n$)")
    plt.ylabel("Competitive Ratio")
plt.legend()
plt.savefig("../figures/all_algos_vary_nsamples.pdf", bbox_inches="tight")
plt.savefig("../figures/all_algos_vary_nsamples.png", bbox_inches="tight")
