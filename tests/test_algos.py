from mlforscheduling.etc_u import etc_u2, etc_u
from mlforscheduling.utils import opt, opt2
import numpy as np


def test_opt_opt2():
    jobs = np.random.rand(2, 5)
    np.testing.assert_allclose(opt(jobs, return_order=(True)), opt2(jobs[0], jobs[1]))


def test_etcu():
    jobs = np.vstack(
        [np.random.exponential(2, size=500), np.random.exponential(20, size=500)]
    )
    jobs = jobs.astype(int)
    order = etc_u(lambda n: 6 * n**2, jobs, return_order=(True))
    assert len(order) == len(jobs.flatten())
    np.testing.assert_allclose(
        etc_u(lambda n: 6 * n**2, jobs, return_order=(True)), etc_u2(jobs[0], jobs[1])
    )


def test_etcu_opt():
    jobs = np.array([[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(etc_u(lambda n: 6 * n**2, jobs, return_order=True), opt(jobs, return_order=(True)))


def test_etcu_worst():
    jobs = np.array([[3, 2, 1], [1, 2, 3]])
    np.testing.assert_allclose(
        etc_u(lambda n: 6 * n**2, jobs, return_order=(True)), np.array([3, 1, 2, 2, 1, 3])
    )


def test_explore():
    n = 200
    jobs = np.vstack(
        [
            np.random.exponential(1, size=n),
            np.random.exponential(10, size=n),
            np.random.exponential(30, size=n),
        ]
    )
    jobs = jobs.astype(int)
    res = etc_u(lambda n: 6 * n**2, jobs, return_type=True)
    print(res)
    assert len(res) == len(jobs.flatten())
