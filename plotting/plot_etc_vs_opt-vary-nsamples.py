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

results = np.load("../experiments/data/etc_vs_opt-vary-nsamples.npy")
# %%
#
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
    plt.semilogx(ns, flow_times[:, i], label=algo, color=COLORS[algo])
    plt.fill_between(
        ns, flow_times_low[:, i], flow_times_high[:, i], alpha=0.1, color=COLORS[algo]
    )
    plt.xlabel("Number of jobs per type ($n$)")
    plt.ylabel("Competitive Ratio")
plt.legend()
plt.savefig("../figures/etc-vs-opt-vary-nsamples.pdf", bbox_inches="tight")

# plt.figure()
# for i, algo in enumerate(["ETC-U", "ETC-RR", "FTPP", "RR"]):
#     # plt.bar(i, flow_times[i], label=algo, alpha=0.5, edgecolor=COLORS[algo], fill=False)
#     plt.scatter(
#         i + np.random.randn(n_seeds) / 10,
#         all_flow_times[:, i],
#         alpha=0.05,
#         color=COLORS[algo],
#     )
#     plt.errorbar(
#         i,
#         flow_times[i],
#         yerr=(
#             [flow_times[i] - flow_times_low[i]],
#             [flow_times_high[i] - flow_times[i]],
#         ),
#         label=algo,
#         color=COLORS[algo]
#     )
# plt.ylabel("CR")
# plt.ylim([1, 2.1])
# plt.legend()
# plt.show()
