"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time
from scipy.stats import chi2
import numba
from numba import njit


@njit
def ucb_rr(jobs,delta=0.001):
    """Explore then commit with uniform exploration.

    UCB in the jobs with confidence bounds of the exponential distribution

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
    return_type=False
    return_order=False
    k, n = jobs.shape
    jobs = jobs.copy()
    m = np.zeros(k,dtype=numba.int64)
    total_time = np.zeros(k,dtype=numba.int64)
    bound = np.ones(k)
    flow_time = 0
    while np.sum(m)<k*n:
        current_type, current_job,duration = select_next_job(bound,jobs,m,total_time,n,k)
        total_time[current_type]+=min(duration,int(current_job/delta)+1)
        if current_job-delta*duration<0.:
            flow_time+= (current_job)*(k*n-np.sum(m))
            jobs[current_type,m[current_type]]-=current_job
            m[current_type]+=1
        else:
            jobs[current_type,m[current_type]]-=delta*duration
            flow_time+= (duration*delta)*(k*n-np.sum(m))
        bound[current_type] = update_confidence_bound(bound,m, current_type , total_time,n,k)

    return flow_time 

@njit
def select_next_job(bound,jobs,m,total_time,n,k):
    current_type = np.argmax(bound)
    current_job = jobs[current_type,m[current_type]]
    duration = select_duration(bound,jobs,m,total_time,current_type,n,k)
    return current_type,current_job,duration

@njit
def select_duration(bound,jobs,m,total_time,current_type,n,k):
    duration = 1
    stop = np.sort(bound)[-2]
    if stop<0:
        return np.inf
    next_bound = update_confidence_bound(bound,m, current_type , total_time+duration,n,k)
    while next_bound>stop:
        duration=2*duration
        next_bound = update_confidence_bound(bound,m, current_type , total_time+duration,n,k)
    return max(1,int(duration/2))



@njit
def update_confidence_bound(bound,m, i , total_time,n,k):
    if m[i]>n-1:
        bound[i]= -np.inf
    else:
        tot= np.sum(total_time)
        mean = m[i]/total_time[i]
        gap = (np.log(k**3*n**2))/(total_time[i])
        bound[i] = klucbBern_numba(mean,gap,lower=False)
    return bound[i]

@njit
def klBern_numba(x, y,eps=1e-9):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

@njit
def klucb_numba(x, d, kl, upperbound=1,
                lowerbound=0, precision=1e-9, max_iterations=200,lower=False):
#lowerbound version
    value = min(x, upperbound)
    if lower:
        u = lowerbound
    else:
        u=upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and np.abs(value - u) > precision:
        _count_iteration += 1
        m = (value + u) / 2.
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2.



@njit
def klucbBern_numba(x, d, precision=1e-9,lower=False):
    upperbound = 1
    return klucb_numba(x, d, klBern_numba, upperbound, precision,lower=lower)
