from plotting.params import COLORS
import matplotlib.pyplot as plt
import numpy as np

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True,
}
plt.rcParams.update(rc)

all_flow_times = np.load("../experiments/data/etc_vs_opt-2-jobs.npy")
flow_times_high = np.quantile(all_flow_times, 0.9, axis=0)
flow_times_low = np.quantile(all_flow_times, 0.1, axis=0)
flow_times = np.median(all_flow_times, axis=0)

lambda1 = np.linspace(0.02, 1, 20)

plt.figure()
for i, algo in enumerate(["ETC-U", "ETC-RR","UCB-U","UCB-RR","RR", "FTPP"]):
    plt.plot(lambda1, flow_times[i], label=algo, color=COLORS[algo])
   # plt.fill_between(
   #     lambda1, flow_times_low[i], flow_times_high[i], alpha=0.3, color=COLORS[algo]
   # )
    plt.axvline(1 / (1 + 2 * np.sqrt(2)), color="black", linestyle="--")
    plt.xlabel("Mean processing time of job 1 ($\lambda_1$)")
    plt.ylabel("Competitive Ratio")
plt.legend()
plt.savefig("../figures/etc-vs-opt-2-jobs.pdf", bbox_inches="tight")
plt.savefig("../figures/etc-vs-opt-2-jobs.png", bbox_inches="tight")
