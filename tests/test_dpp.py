import unittest
import numpy as np
from detectadvprompt import dp


class TestDPPFunction(unittest.TestCase):
    def test_1(self):
        a_s = np.array([0, 0, 0, 0, 0, 0])
        lambdas = np.array([0.1, 0.2, 0.3])
        ps = np.array([dp.dp_prob(a_s, lamb, 0.0) for lamb in lambdas])
        self.assertTrue(np.allclose(ps, np.ones((3, 6)) / 2))
        ps2 = dp.dp_prob(a_s, lambdas, 0.0)
        self.assertTrue(np.allclose(ps, ps2))

    def test_2(self):
        a_s = np.array([1, 1, 1, 1, 1, 1])
        lambdas = np.array([0.1, 0.2, 0.3])
        ps = np.array([dp.dp_prob(a_s, lamb, 0.0) for lamb in lambdas])
        self.assertTrue(np.allclose(ps, np.flip(ps, axis=-1)))
        ps2 = dp.dp_prob(a_s, lambdas, 0.0)
        self.assertTrue(np.allclose(ps, ps2))

    def test_3(self):
        a_s = np.array([1, 2, 3, 4, 5, 6])
        lambdas = np.array([0.1, 0.2, 0.3])
        ps = np.array([dp.dp_prob(a_s, lamb, 0.0) for lamb in lambdas])
        self.assertTrue(np.all(ps[:, -1] < ps[:, 0]))
        ps2 = dp.dp_prob(a_s, lambdas, 0.0)
        self.assertTrue(np.allclose(ps, ps2))
