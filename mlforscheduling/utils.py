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


def opt(jobs):
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
    order = jobs.flatten()
    order = np.sort(order)
    return order


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
