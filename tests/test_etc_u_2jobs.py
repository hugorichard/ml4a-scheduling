from mlforscheduling.etc_u import etc_u
from mlforscheduling.etc_u import opt
import numpy as np


def test_etcu_opt():
    jobs1 = np.array([1, 2, 3])
    jobs2 = np.array([1, 2, 3])
    np.testing.assert_allclose(etc_u(jobs1, jobs2), opt(jobs1, jobs2))


def test_etcu_worst():
    jobs1 = np.array([3, 2, 1])
    jobs2 = np.array([1, 2, 3])
    np.testing.assert_allclose(etc_u(jobs1, jobs2),np.array([3, 1, 2, 2, 1, 3]))


def test_explore():
    jobs1 = 0.1 * np.ones(50) + np.random.rand(50)
    jobs2 = 0.5 * np.ones(50) + np.random.rand(50)
    print(etc_u(jobs1, jobs2))
    assert False
