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
    delta=1
    f=lambda n: 6 * n**2
    return_type=False
    return_order=False
    k, n = jobs.shape
    m = np.zeros(k,dtype=numba.int64)
    total_time = np.zeros(k,dtype=numba.int64)
    bound = np.zeros(k)
    flow_time = 0
    while np.sum(m)<k*n:
        current_type, current_job,duration = select_next_job(bound,jobs,m)
        total_time[current_type]+=duration
        if current_job-delta*duration<0.:
            print("comparison",delta,jobs[current_type,m[current_type]] )
            flow_time+= (current_job)*(k*n-np.sum(m))*100
            jobs[current_type,m[current_type]]-=current_job
            print("job finished",jobs[current_type,m[current_type]] )
            m[current_type]+=1
        else:
            jobs[current_type,m[current_type]]-=delta*duration
            flow_time+= (duration*delta)*(k*n-np.sum(m))
        bound =update_confidence_bound(bound,m, current_type , total_time,n,k)
    print(np.sum(jobs),np.sum(m) )
    return flow_time 

@njit
def select_next_job(bound,jobs,m):
    current_type = np.argmax(bound)
    current_job = jobs[current_type,m[current_type]]
    duration = select_duration(bound,jobs,m)
    return current_type,current_job,1

@njit
def select_duration(bound,jobs,m):
    duration = 1

    return duration



@njit
def update_confidence_bound(bound,m, i , total_time,n,k):
    if m[i]>n-1:
        bound[i]= -np.inf
    else:
        tot= np.sum(total_time)
        mean = m[i]/total_time[i]
        gap = (np.log(k**3*n**2))/(tot)
        bound[i] = klucbBern_numba(mean,gap,lower=False)
    return bound

@njit
def klBern_numba(x, y,eps=1e-6):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

@njit
def klucb_numba(x, d, kl, upperbound=1,
                lowerbound=0, precision=1e-6, max_iterations=100,lower=False):
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
    #print(_count_iteration )
    return (value + u) / 2.



@njit
def klucbBern_numba(x, d, precision=1e-6,lower=False):
    upperbound = 1.#min(1., klucbGauss_numba(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    lowerbound = 0.
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb_numba(x, d, klBern_numba, upperbound, precision,lower=lower)


#@njit
#def update_confidence_bound2(bound,m, i , total_time,n):
#    if m[i]>n-1:
#        bound[i]= -np.inf
#    else:
#        tot= np.sum(total_time)
#        mean = m[i]/total_time[i]
#        var = (m[i]*(1-mean)**2+(total_time[i]-m[i])*mean**2)/total_time[i]
#        bound[i] = mean+np.sqrt(4*var*np.log(tot)/total_time[i])+2*np.log(tot)/total_time[i]
#    return 

