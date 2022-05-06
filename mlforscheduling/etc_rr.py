"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import rr_run, flow_time


def etc_rr(f, jobs, return_type_done=False):
    """Explore then commit with RR exploration.

    Explore simultaneously all types that are likely to be the smallest.

    Parameters
    ----------
    f: int -> int
        A function of n

    jobs : np array of size k, n
        jobs[i, j] is the processing times of the jth job of type i

    Return
    ------
    flow_time : float
        Flow time obtained
    """
    k, n = jobs.shape
    alpha = np.zeros(k)
    o = np.zeros((k, k))
    U = np.arange(k)
    d = np.zeros((k, k))
    d = d + np.inf
    r = np.zeros((k, k))
    r = r + 0.5
    J = np.zeros(k)
    time = 0
    f_time = 0
    type_done = []
    remaining_jobs = np.copy(jobs)
    for _ in range(n*k):
        A = []
        for z in U:
            keep = True
            for i in U:
                if i == z:
                    continue
                if r[i, z] - d[i, z] > 0.5:
                    keep = False
            if keep:
                A.append(z)
        A = np.array(A)
        A = A.astype(np.int_)
        J = J.astype(np.int_)
        if len(A) > 1:
            time, remaining_jobs[A, J[A]], alpha = rr_run(
                time, remaining_jobs[A, J[A]]
            )
            alpha = A[alpha]
            type_done.append(alpha)
            J[alpha] += 1
            f_time += time

            for z in A:
                if z == alpha:
                    continue
                o[alpha, z] += 1
                d[alpha, z] = np.sqrt(
                    np.log(2 * f(n)) / (2 * (o[alpha, z] + o[z, alpha]))
                )
                r[alpha, z] = o[alpha, z] / (o[alpha, z] + o[z, alpha])

            if J[alpha] >= n:
                U = [u for u in U if u != alpha]
        else:
            a = A[0]
            rja = remaining_jobs[a, :]
            rja = rja[rja > 0]
            rja[0] = rja[0] + time
            f_time += flow_time(rja)
            for _ in range(len(rja)):
                type_done.append(a)
            U = [u for u in U if u != a]
        if len(U) == 0:
            break

    if return_type_done:
        return type_done
    else:
        return f_time
