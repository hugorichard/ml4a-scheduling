"""Reproducing Figure 2: plotting"""
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

all_flow_times = np.load("../experiments/data/n_50_vary_lambdas_logscale.npy")
print(np.shape(all_flow_times))
n, _, n_lambda = np.shape(all_flow_times)
flow_times = np.mean(all_flow_times, axis=0)
flow_times_low = np.zeros_like(flow_times)
flow_times_high = np.zeros_like(flow_times)

deviation = np.std(all_flow_times, axis=0) / np.sqrt(n)

for i in range(6):
    for j in range(n_lambda):
        flow_times[i, j] = (flow_times[i, j] - flow_times[-2, j]) / flow_times[-1, j]
        flow_times_low[i, j] = (
            flow_times[i, j] - 2 * deviation[i, j] / flow_times[-1, j]
        )
        flow_times_high[i, j] = (
            flow_times[i, j] + 2 * deviation[i, j] / flow_times[-1, j]
        )

lambda1 = [0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]
plt.figure()
plt.rcParams["text.usetex"] = True
plt.ylim(top=5, bottom=10 ** (-4))
for i, algo in enumerate(["ETC-U", "ETC-RR", "UCB-U", "UCB-RR", "RR", "FTPP"]):
    if algo != "FTPP":
        plt.yscale("log")
        plt.semilogx(lambda1, flow_times[i], label=algo, color=COLORS[algo])
        # plt.plot(lambda1, flow_times[i], label=algo, color=COLORS[algo])
        plt.fill_between(
            lambda1,
            flow_times_low[i],
            flow_times_high[i],
            alpha=0.3,
            color=COLORS[algo],
        )
        # plt.plot(lambda1,lambda1,'o')
        # plt.axvline(1 / (1 + 2 * np.sqrt(2)), color="black", linestyle="--")
        plt.xlabel("Mean processing time of job 1 ($\lambda_1$)")
        plt.ylabel(r"$(E[C_A]-E[C_{FTPP}])/E[C_{OPT}]$")
plt.legend()
plt.savefig("../figures/n_50_vary_lambda_logscale.pdf", bbox_inches="tight")
plt.savefig("../figures/n_50_vary_lambda_logscale.png", bbox_inches="tight")
