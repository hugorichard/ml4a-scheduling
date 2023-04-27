from mlforscheduling.etc_u import etc_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.utils import opt, rr, rr_run, ftpp
from mlforscheduling.lsept import lsept
from mlforscheduling.ucb_u import ucb_u
from mlforscheduling.ucb_rr import ucb_rr
import numpy as np
import pytest


def test_rr():
    jobs = np.random.randint(10, size=10) + 1 + np.random.rand(10)
    cost = rr(jobs)
    n = len(jobs)
    current_time = 0
    new_jobs = jobs
    costrr = 0
    for i in range(n):
        current_time, new_jobs, i = rr_run(current_time, new_jobs)
        costrr += current_time
    assert np.abs(costrr - cost) < 1e-10



def test_etcu_opt():
    jobs = np.array([[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(
        etc_u(jobs, return_order=True),
        opt(jobs, return_order=(True)),
    )


def test_etcu_worst():
    jobs = np.array([[3, 2, 1], [1, 2, 3]])
    np.testing.assert_allclose(
        etc_u(jobs, return_order=(True)),
        np.array([3, 1, 2, 2, 1, 3]),
    )


def test_etcu_explore():
    n = 200
    jobs = np.vstack(
        [
            np.random.exponential(1, size=n),
            np.random.exponential(10, size=n),
            np.random.exponential(30, size=n),
        ]
    )
    jobs = jobs.astype(int)
    res = etc_u(jobs, return_type=True)
    print(res)
    assert len(res) == len(jobs.flatten())


def test_ftpp():
    n = 5
    jobs = np.array([[1] * n, [2] * n, [4] * n])
    f = ftpp(jobs)
    f2 = opt(jobs)
    print(4 * n )
    assert f == f2


def test_etcrr_commit():
    n = 3
    jobs = np.vstack([[10] * n + np.arange(n), [100] * n + np.arange(n)])
    flow1 = etc_rr(jobs, f=lambda n: 1 / 5 * n)
    res = 2 * 10
    res += 2 * 10 + 11
    res += 2 * 10 + 11 + 12
    res += 2 * 10 + 11 + 12 + 90
    res += 2 * 10 + 11 + 12 + 90 + 101
    res += 2 * 10 + 11 + 12 + 90 + 101 + 102
    assert np.abs(flow1 - res) < 1e-10


def test_etcu_explore():
    n = 200
    jobs = np.vstack(
        [
            np.random.exponential(1, size=n),
            np.random.exponential(10, size=n),
            np.random.exponential(30, size=n),
        ]
    )
    res = etc_rr(jobs)
    res2 = etc_u(jobs)
    assert res < res2


@pytest.mark.parametrize("algo", [lsept, etc_u, etc_rr, ucb_u, ucb_rr,rr, ftpp,opt])
def test_algos(algo):
    n = 1000
    jobs = np.vstack(
        [
            np.random.exponential(1.8, size=n),
            np.random.exponential(2, size=n),
            np.random.exponential(4, size=n),
        ]
    )
    algo(jobs)