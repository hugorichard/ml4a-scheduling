"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time
from scipy.stats import chi2
import numba
from numba import njit


@njit
def ucb_rr(jobs):
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
    delta=0.1
    f=lambda n: 6 * n**2
    return_type=False
    return_order=False
    k, n = jobs.shape
    m = np.zeros(k,dtype=numba.int64)
    total_time = np.zeros(k,dtype=numba.int64)
    bound = np.zeros(k)
    flow_time = 0
    while np.sum(m)<k*n:
        current_type, current_job = select_next_job(bound,jobs,m)
        total_time[current_type]+=1
        if current_job-delta<0.:
            m[current_type]+=1
            flow_time+= (delta-current_job)*(k*n-np.sum(m))
        else:
            jobs[current_type,m[current_type]]-=delta
            flow_time+= delta*(k*n-np.sum(m))
        bound =update_confidence_bound(bound,m, current_type , total_time,n)
    return flow_time

@njit
def select_next_job(bound,jobs,m):
    current_type = np.argmax(bound)
    current_job = jobs[current_type,m[current_type]]
    return current_type,current_job

@njit
def update_confidence_bound(bound,m, i , total_time,n):
    if m[i]>n-1:
        bound[i]= -np.inf
    else:
        tot= np.sum(total_time)
        mean = m[i]/total_time[i]
        var = (m[i]*(1-mean)**2+(total_time[i]-m[i])*mean**2)/total_time[i]
        bound[i] = mean+np.sqrt(4*var*np.log(tot)/total_time[i])+2*np.log(tot)/total_time[i]
    return bound
