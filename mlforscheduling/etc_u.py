"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time


def etc_u2(jobs1, jobs2):
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
        if m > (n - 1):
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


def etc_u(f, jobs, return_type=False, return_order=False):
    """Explore then commit with uniform exploration.

    Explore jobs alternatively and commit to best options when confident enough.
    All jobs of the same type are assumed to follow an exponential distribution
    with the same mean.

    Parameters
    ----------
    f: int -> int
        A function of n

    jobs : np array of size k, n
        jobs[i, j] is the processing times of the jth job of type i

    Return
    ------
    order : np array of size kn
        The processing times ordered as executed by the algo
    """
    # Assume the jobs have the same length
    order = []
    type_order = []
    k, n = jobs.shape
    m = np.ones(k)
    P = np.zeros((k, n))
    P[:, 0] = jobs[:, 0]
    for t, job in enumerate(jobs[:, 0]):
        order.append(job)
        type_order.append(t)
    d = np.zeros((k, k))
    r = np.zeros((k, k))
    U = []
    for i in range(k):
        if m[i] < n:
            U.append(i)

    for _ in range(k * n):
        for i in U:
            for j in U:
                if i == j:
                    continue
                m_ij = int(min(m[i], m[j]))
                d[i, j] = np.sqrt(np.log(2 * f(n)) / (2 * m_ij))
                r[i, j] = 1 / m_ij * np.sum(P[i, :m_ij] < P[j, :m_ij])


        A = []
        for z in U:
            addz = True
            for i in U:
                if i == z:
                    continue
                if r[i, z] - d[i, z] > 0.5:
                    addz = False
            if addz:
                A.append(z)
        A = np.array(A)
        if len(A) > 1:
            j = np.argmin(m[A])
            j = A[j]
            m[j] += 1
            P[j, int(m[j]) - 1] = jobs[j, int(m[j]) - 1]
            order.append(jobs[j, int(m[j]) - 1])
            type_order.append(j)
            if m[j] >= n:
                U = [z for z in U if z != j]
        else:
            j = A[0]
            P[j, int(m[j]) :] = jobs[j, int(m[j]) :]
            for job in jobs[j, int(m[j]) :]:
                order.append(job)
                type_order.append(j)
            m[j] += n - m[j]
            U = [z for z in U if z != j]
        if len(U) == 0:
            if return_order:
                return np.array(order)
            if return_type:
                return np.array(type_order)
            return flow_time(order)
