"""Functions for scheduling based on ML predictions."""
import numpy as np


def rr(jobs):
    """Compute the flow time of Round Robin.

    Parameters
    ----------
    jobs : np array of size k, n
        jobs[i, j] is the processing times of the jth job of type i

    Return
    ------
    flow_time: float
        The flow time
    """
    X = jobs.flatten()
    n = len(X)
    return (2 * np.flip(np.arange(n)) + 1) @ np.sort(X)


def rr_run(current_time, old_jobs):
    """Flow time completing one job with RR.

    Parameters
    ------
    current_time: float
    old_jobs: np array

    Return
    ------
    new_current_time: float
    flow_time: float
    new_jobs: np array
    i: index of finished job
    """
    jobs = np.copy(old_jobs)
    I = jobs > 0
    jobs = jobs[I]
    n = len(jobs)
    i = np.argmin(jobs)
    ji = jobs[i]
    assert np.sum(jobs == ji) < 2  # Check that no jobs are equal
    current_time += n * ji
    return current_time, old_jobs - ji, i


def ftpp(jobs):
    """Follow the perfect prediction.
    Computes jobs in order of their increasing type means.

    Parameters
    ----------

    jobs: np array of size (k, n)
        Must be ordered by increasing types means
    """
    k, _ = np.shape(jobs)
    I = np.arange(k)
    order = jobs[I, :]
    return flow_time(order.flatten())


def opt(jobs, return_type=False, return_order=False):
    """Optimal scheduling.

    Schedule the smallest job first

    Parameters
    ----------
    jobs : np array of size k, n
        jobs[i, j] is the processing times of the jth job of type i

    Return
    ------
    order: np array of size kn
        The processing times ordered as executed by the algo
    """
    k, n = jobs.shape
    order = jobs.flatten()
    order = np.sort(order)
    if return_order:
        return order
    if return_type:
        return np.array([[i] * n for i in range(k)])
    return flow_time(order)


def flow_time(order):
    """Compute the flow time of some ordering.

    The jobs are executed in the order given by the array
    Parameters
    ----------
    order: np array

    Return
    ------
    flow_time : float
        Total flow time
    """
    current_time = 0
    flow_times = []
    for p in order:
        current_time += p
        flow_times.append(current_time)
    return np.sum(flow_times)


from numba import jit
import numpy as np


# @jit(nopython=True)
def klBern_numba(x, y, eps=1e-6):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


# @jit
def klucb_numba(
    x,
    d,
    kl,
    upperbound=1,
    lowerbound=0,
    precision=1e-6,
    max_iterations=100,
    lower=False,
):
    # lowerbound version
    value = min(x, upperbound)
    if lower:
        u = lowerbound
    else:
        u = upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and np.abs(value - u) > precision:
        _count_iteration += 1
        m = (value + u) / 2.0
        if kl(x, m) > d:
            u = m
        else:
            value = m
    # print(_count_iteration )
    return (value + u) / 2.0


# @jit
def klucbBern_numba(x, d, precision=1e-6, lower=False):
    upperbound = 1.0  # min(1., klucbGauss_numba(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    lowerbound = 0.0
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb_numba(x, d, klBern_numba, upperbound, precision, lower=lower)
