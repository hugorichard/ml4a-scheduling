"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time
from scipy.stats import chi2


def etc_u(jobs, f=lambda n: 6 * n**2, return_order=False):
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

    return_order : bool
        If True, etc_u returns order, If False, returns flow_time

    Return
    ------
    order: np array
        Array of job sizes ordered by starting time

    flow_time : float
        Flow time obtained
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
            return flow_time(order)
