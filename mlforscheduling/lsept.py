"""Functions for bayesian scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time


def lsept(jobs, alpha=2, w=0, return_type=False, return_order=False):
    """LSEPT bayesian approach of S Marban

    Use the learning lsept

    Parameters
    ----------
    jobs : np array of size k, n
        jobs[i, j] is the processing times of the jth job of type i

    alpha : float
        Parameter of the Gamma prior for all types

    w : float
        Parameter of the Gamma prior for all types

    Return
    ------
    order : np array of size kn
        The processing times ordered as executed by the algo
    """
    # Assume the jobs have the same length
    order = []
    type_order = []
    k, n = jobs.shape
    alphas = np.zeros(k)
    ws = np.zeros(k)
    alphas += alpha
    ws += w

    criterion = ws / (alphas - 1)
    indexes = np.zeros(k)
    sum_jobs = np.zeros(k)
    n_jobs = np.zeros(k)

    for _ in range(k * n):
        imin = np.argmin(criterion)
        j = int(indexes[imin])
        job = jobs[imin, j]
        indexes[imin] += 1
        order.append(job)
        type_order.append(imin)
        sum_jobs[imin] += job
        n_jobs[imin] += 1
        if indexes[imin] >= n:
            criterion[imin] = np.inf
        else:
            criterion[imin] = (ws[imin] + sum_jobs[imin]) / (
                alphas[imin] + n_jobs[imin] - 1
            )

    if return_order:
        return np.array(order)
    if return_type:
        return np.array(type_order)
    return flow_time(order)
