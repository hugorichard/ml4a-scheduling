"""Functions for scheduling based on ML predictions."""
import numpy as np


def etc_u(jobs1, jobs2):
    """Explore then commit with uniform exploration.

    Explore jobs alternatively and commit to best options when confident enough.
    All jobs of the same type are assumed to follow an exponential distribution
    with the same mean.

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
    # Assume the jobs have the same length
    assert len(jobs1) == len(jobs2)

    n = len(jobs1)
    m = 0
    current_time = 0
    flow_times = []
    remaining_jobs1 = np.full(n, True)
    remaining_jobs2 = np.full(n, True)
    order = []
    r_hat = 0
    while True:
        # Run job of type 1
        p1 = jobs1[m]
        order.append(p1)
        remaining_jobs1[m] = False
        # Run job of type 2
        p2 = jobs2[m]
        order.append(p2)
        remaining_jobs2[m] = False
        r_hat = (r_hat * m + int(p1 < p2)) / (m + 1)
        delta = np.sqrt(np.log(12 * n**2) / (2 * (m + 1)))
        if r_hat - delta > 0.5 or r_hat + delta < 0.5:
            break
        m += 1
        if m > (n-1):
            break

    if r_hat > 0.5:
        job_list = [jobs1[remaining_jobs1], jobs2[remaining_jobs2]]
    else:
        job_list = [jobs2[remaining_jobs2], jobs1[remaining_jobs1]]

    # jobs of type 1 are lower
    for jobs in job_list:
        for p in jobs:
            order.append(p)
    return np.array(order)


def opt(jobs1, jobs2):
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
