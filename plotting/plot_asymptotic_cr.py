import matplotlib.pyplot as plt
import numpy as np

k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
opt_list = np.load("../experiments/data/asympotic-cr-opt_list.npy")
approx_list = np.load("../experiments/data/asympotic-cr-approx_list.npy")
plt.rc("font", size=15)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(np.arange(1, 1000 + 1), approx_list, label=r"$CR_{FTPP}(\tilde{\lambda})$")
ax.plot([1] + k_range, opt_list, marker="o", label="Numerical Optimization Result")
ax.set_xscale("log")
ax.axhline(2, label="CR RR", linestyle="--", color="black")
ax.axhline(1, color="black")
ax.set_xlabel("K")
ax.set_ylabel("CR")
ax.legend()
plt.savefig("../figures/asymptotic-cr.pdf", bbox_inches="tight")
plt.savefig("../figures/asymptotic-cr.png", bbox_inches="tight")
