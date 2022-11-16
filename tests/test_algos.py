from mlforscheduling.etc_u import etc_u2, etc_u
from mlforscheduling.etc_rr import etc_rr
from mlforscheduling.utils import opt, opt2, rr, rr_run, ftpp, rr_per_type
from mlforscheduling.lsept import lsept
import numpy as np


def test_opt_opt2():
    jobs = np.random.rand(2, 5)
    np.testing.assert_allclose(opt(jobs, return_order=(True)), opt2(jobs[0], jobs[1]))


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


def test_etcrr2():
    n = 3
    jobs = np.vstack(
        [[10] * n + np.arange(n), [20] * n + np.arange(n), [30] * n + np.arange(n)]
    )
    print(jobs)
    f_time = etc_rr(jobs)
    cost = 10 * 3  # 11, 10, 20, Done [0]
    cost += 10 * 3 + 10 * 3  # 1, 21, 10, Done [0, 1]
    cost += 10 * 3 + 10 * 3 + 1 * 3  # 12, 20, 9, Done [0, 1, 0]
    cost += 10 * 3 + 10 * 3 + 1 * 3 + 9 * 3  # 3, 11, 31, Done [0, 1, 0, 2]
    cost += 10 * 3 + 10 * 3 + 1 * 3 + 9 * 3 + 3 * 3  # 8, 28, Done [0, 1, 0, 2, 0]
    cost += (
        10 * 3 + 10 * 3 + 1 * 3 + 9 * 3 + 3 * 3 + 8 * 2
    )  # 22, 20, Done [0, 1, 0, 2, 0, 1]
    cost += (
        10 * 3 + 10 * 3 + 1 * 3 + 9 * 3 + 3 * 3 + 8 * 2 + 20 * 2
    )  # 2, 32, Done [0, 1, 0, 2, 0, 1, 2]
    cost += 10 * 3 + 10 * 3 + 1 * 3 + 9 * 3 + 3 * 3 + 8 * 2 + 20 * 2 + 2 * 2
    # 30, Done [0, 1, 0, 2, 0, 1, 2, 1]
    cost += 10 * 3 + 10 * 3 + 1 * 3 + 9 * 3 + 3 * 3 + 8 * 2 + 20 * 2 + 2 * 2 + 30
    cost2 = rr_per_type(jobs)
    assert cost == f_time
    assert cost == cost2


def test_etcu():
    jobs = np.vstack(
        [np.random.exponential(2, size=500), np.random.exponential(20, size=500)]
    )
    jobs = jobs.astype(int)
    order = etc_u(jobs, return_order=(True))
    assert len(order) == len(jobs.flatten())
    np.testing.assert_allclose(
        etc_u(jobs, return_order=(True)), etc_u2(jobs[0], jobs[1])
    )


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
    jobs = np.array([[4] * n, [2] * n, [1] * n])
    f = ftpp(jobs)
    f2 = opt(jobs)
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


def test_lsept():
    n = 1000
    jobs = np.vstack(
        [
            np.random.exponential(1.8, size=n),
            np.random.exponential(2, size=n),
        ]
    )
    lsept(jobs, alpha=2, w=0)
    assert False
