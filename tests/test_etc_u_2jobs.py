from mlforscheduling.etc_u import etc_u2, etc_u
from mlforscheduling.utils import opt, opt2
import numpy as np


def test_opt_opt2():
    jobs = np.random.rand(2, 5)
    np.testing.assert_allclose(opt(jobs), opt2(jobs[0], jobs[1]))

def test_etcu():
    jobs = np.vstack([np.random.exponential(2, size=500), np.random.exponential(20, size=500)])
    jobs = jobs.astype(int)
    print(jobs.shape)
    order = etc_u(lambda n: 6 * n ** 2, jobs)
    print(order)
    assert len(order) == len(jobs.flatten())
    np.testing.assert_allclose(etc_u(lambda n: 6 * n**2, jobs), etc_u2(jobs[0], jobs[1]))
    assert False

def test_etcu_opt():
    jobs1 = np.array([1, 2, 3])
    jobs2 = np.array([1, 2, 3])
    np.testing.assert_allclose(etc_u(jobs1, jobs2), opt(jobs1, jobs2))


def test_etcu_worst():
    jobs1 = np.array([3, 2, 1])
    jobs2 = np.array([1, 2, 3])
    np.testing.assert_allclose(etc_u(jobs1, jobs2), np.array([3, 1, 2, 2, 1, 3]))


def test_explore():
    jobs1 = 0.1 * np.ones(50) + np.random.rand(50)
    jobs2 = 0.5 * np.ones(50) + np.random.rand(50)
    print(etc_u(jobs1, jobs2))
    assert False
