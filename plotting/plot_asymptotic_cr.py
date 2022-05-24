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

k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
opt_list = np.load("../experiments/data/asympotic-cr-opt_list.npy")
approx_list = np.load("../experiments/data/asympotic-cr-approx_list.npy")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1, 1000 + 1), approx_list, label=r"$CR_{FTPP}(\tilde{\lambda})$")
ax.plot([1] + k_range, opt_list, marker="o", label="Numerical Optimization Result")
ax.set_xscale("log")
ax.axhline(2, label="RR", linestyle="--", color=COLORS["RR"])
ax.axhline(1, color="black")
ax.set_xlabel("Number of types ($K$)")
ax.set_ylabel("Competitive Ratio")
ax.legend()
plt.savefig("../figures/asymptotic-cr.pdf", bbox_inches="tight")
plt.savefig("../figures/asymptotic-cr.png", bbox_inches="tight")
