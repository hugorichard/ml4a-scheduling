"""Reproducing Figure 3: run experiment."""
import numpy as np
from scipy.optimize import LinearConstraint, minimize


def CR(x):
    K = len(x)
    num = 0
    denum = 0
    for i in range(K):
        num += (K - (i + 1) + 1 / 2) * x[i]
        denum += x[i] / 4
        for j in range(i + 1, K):
            denum += x[i] * x[j] / (x[i] + x[j])
    return num / denum


def phi(x):
    return CR(np.append(x, 1))


opt_list = [2]
k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
for k_types in k_range:
    bounds = [(0, 1)] * (k_types - 1)
    A = np.diag(-np.ones(k_types - 2), -1) + np.eye(k_types - 1)
    lin_cons = LinearConstraint(A, np.zeros(k_types - 1), np.inf * np.ones(k_types - 1))
    x_0 = np.flip(1 / np.arange(2, k_types + 1) ** 2)
    opt_res = minimize(phi, x_0, bounds=bounds, constraints=(lin_cons,))
    if not opt_res.success:
        print("Optimization Failure")
    opt_list.append(opt_res.fun)

approx_list = []
for k_types in np.arange(1, 1001):
    x_0 = np.flip(1 / np.arange(2, k_types + 1) ** 2)
    approx_list.append(phi(x_0))

np.save("./data/asympotic-cr-opt_list.npy", opt_list)
np.save("./data/asympotic-cr-approx_list.npy", approx_list)
