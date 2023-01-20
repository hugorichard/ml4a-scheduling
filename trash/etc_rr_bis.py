"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import rr_run, flow_time, klucbBern_numba


def etc_rr_bis(jobs, f=lambda n: 6 * n**2, return_type_done=False):
    k, n = jobs.shape
    alpha = np.zeros(k)
    o = np.zeros((k, k))
    U = np.arange(k)
    d = np.zeros((k, k))
    d = d + np.inf
    r = np.zeros((k, k))
    r = r + 0.5
    lb = np.zeros((k, k))
    J = np.zeros(k)
    time = 0
    f_time = 0
    type_done = []
    remaining_jobs = np.copy(jobs)
    for i in range(n * k):
        A = []
        for z in U:
            keep = True
            for i in U:
                if i == z:
                    continue
                if compute_bound(o, i,z,n) > 0.5:
                    keep = False
            if keep:
                A.append(z)
        A = np.array(A)
        A = A.astype(np.int_)
        J = J.astype(np.int_)
        if len(A) > 1:
            time, remaining_jobs[A, J[A]], alpha = rr_run(time, remaining_jobs[A, J[A]])
            alpha = A[alpha]
            type_done.append(alpha)
            J[alpha] += 1
            f_time += time

            for z in A:
                if z == alpha:
                    continue
                o[alpha, z] += 1
                r[alpha, z] = o[alpha, z] / (o[alpha, z] + o[z, alpha])
                #lb[alpha,z] = compute_bound(o, alpha,z)

            if J[alpha] >= n:
                U = [u for u in U if u != alpha]
        else:
            if np.max(J)<n:
                pass# print(i,"commit", A[0])
            a = A[0]
            rja = remaining_jobs[a, :]
            rja = rja[rja > 0]
            dt = np.sum(rja)
            rja[0] = rja[0] + time
            f_time += flow_time(rja)
            time = time + dt
            remaining_jobs[a, :]= 0
            for _ in range(len(rja)):
                type_done.append(a)
            U = [u for u in U if u != a]
        if len(U) == 0:
            break

    if return_type_done:
        return type_done
    else:
        return f_time

def compute_bound(o, alpha,z,n,printing=False):
    t = np.sum(o)
    if (o[alpha, z] + o[z, alpha])==0 or t<2.:
        return 0.
    bound = (np.log(n)+3*np.log(np.log(n)))/(o[alpha, z] + o[z, alpha])
    r = o[alpha, z] / (o[alpha, z] + o[z, alpha])
    if printing:
        print(r,bound, klucbBern_numba(r,bound),o)
    return klucbBern_numba(r,bound,lower=True)



