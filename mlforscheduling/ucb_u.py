"""Functions for scheduling based on ML predictions."""
import numpy as np
from mlforscheduling.utils import flow_time
from scipy.stats import chi2

def ucb_u(jobs, f=lambda n: 6 * n**2, return_type=False, return_order=False):
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
    order = []
    type_order = []
    k, n = jobs.shape
    m = np.zeros(k,dtype=int)
    total_time = np.zeros(k)
    bound = np.zeros(k)
    for _ in range(k*n):
        current_type, current_job = select_next_job(bound,jobs,m)
        m[current_type]+=1
        total_time[current_type]+= current_job
        bound =update_confidence_bound(bound,m, current_type , total_time,n,k)
        order.append(current_job)
        type_order.append(current_type)
    if return_order:
        return np.array(order)
    if return_type:
        return np.array(type_order)
    return flow_time(order)
            

def select_next_job(bound,jobs,m):
    current_type = np.argmin(bound)
    current_job = jobs[current_type,m[current_type]]
    return current_type,current_job

def update_confidence_bound(bound,m, i , total_time,n,k):
    if m[i]>n-1:
        bound[i]= float('inf')
    else:
        bound[i] = 2*total_time[i]/chi2.ppf(1-1/(2*k**2*n**2),2*m[i])
    return bound



