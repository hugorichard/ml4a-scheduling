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

ns = np.logspace(start=2, stop=6, num=10)

results = np.load("../experiments/data/vary-nsamples_lsept.npy")
# %%
#
all_flow_times = np.array(results)
n_seeds = len(all_flow_times)
deviation = np.std(all_flow_times, axis=0)
flow_times = np.mean(all_flow_times, axis=0)

indices = {}
for i, algo in enumerate(["RR","FTPP","LSEPT"]):#"ETC-U", "ETC-RR", "UCB-U","UCB-RR", 
    indices[algo] = i

plt.figure()
for ii, algo in enumerate([ "RR","FTPP","LSEPT"]):#"ETC-U", "ETC-RR", "UCB-U","UCB-RR",
    i = indices[algo]
    plt.semilogx(ns, flow_times[:, i], label=algo, color=COLORS[algo])
    plt.fill_between(ns, flow_times[:, i]-2*deviation[:, i]/np.sqrt(n_seeds), flow_times[:, i]+2*deviation[:, i]/np.sqrt(n_seeds), alpha=0.1, color=COLORS[algo])
    plt.xlabel("Number of jobs per type ($n$)")
    plt.ylabel("Competitive Ratio")
plt.legend()
plt.savefig("../figures/vary-nsamples_lsept.pdf", bbox_inches="tight")
plt.savefig("../figures/vary-nsamples_lsept.png", bbox_inches="tight")
