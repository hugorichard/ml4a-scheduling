"""Functions for scheduling based on ML predictions."""
import numpy as np


def opt2(jobs1, jobs2):
    """Optimal scheduling.

    Schedule the smallest job first

    Parameters
    ----------
    jobs1 : np array of size n
        Jobs processing times of jobs of type 1

    jobs2 : np array of size n
        Jobs processing times of jobs of type 2

    Return
    ------
    order : np array of size 2n
        The processing times ordered as executed by the algo
    """
    jobs = list(jobs1) + list(jobs2)
    order = np.sort(jobs)
    return order


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


def rr_per_type(jobs):
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
    k, n = jobs.shape
    J = np.zeros(k).astype(np.int_)
    U = np.arange(k)
    f_time = 0
    time = 0
    type_done = []
    remaining_jobs = np.copy(jobs)
    for _ in range(n * k):
        time, remaining_jobs[U, J[U]], alpha = rr_run(time, remaining_jobs[U, J[U]])
        alpha = U[alpha]
        type_done.append(alpha)
        J[alpha] += 1
        f_time += time
        if J[alpha] >= n:
            U = [u for u in U if u != alpha]
        if len(U) == 0:
            break
    return f_time


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
    """Follow the perfect predictions."""
    means = np.mean(jobs, axis=1)
    I = np.argsort(means)
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
