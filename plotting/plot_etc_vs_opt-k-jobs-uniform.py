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

ks = np.arange(2, 11)

results = np.load("../experiments/data/etc_vs_opt-k-jobs-uniform.npy")
all_flow_times = np.array(results)
n_seeds = len(all_flow_times)

flow_times_high = np.quantile(all_flow_times, 0.9, axis=0)
flow_times_low = np.quantile(all_flow_times, 0.1, axis=0)
flow_times = np.median(all_flow_times, axis=0)

indices = {}
for i, algo in enumerate(["ETC-U", "ETC-RR", "FTPP", "RR"]):
    indices[algo] = i

plt.figure()
for ii, algo in enumerate(["ETC-U", "ETC-RR", "RR", "FTPP"]):
    i = indices[algo]
    plt.plot(ks, flow_times[:, i], label=algo, color=COLORS[algo])
    plt.fill_between(
        ks, flow_times_low[:, i], flow_times_high[:, i], alpha=0.1, color=COLORS[algo]
    )
    plt.xlabel("Number of types ($K$)")
    plt.ylabel("Competitive Ratio")
plt.legend()
plt.savefig("../figures/etc-vs-opt-k-jobs-uniform.pdf", bbox_inches="tight")
plt.savefig("../figures/etc-vs-opt-k-jobs-uniform.pdf", bbox_inches="tight")
