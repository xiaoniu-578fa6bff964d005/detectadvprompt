import unittest
import numpy as np
from detectadvprompt import dp


class TestDPPFunction(unittest.TestCase):
    def test_1(self):
        a_s = np.array([1, 2, 3, 4, 5, 6])
        lambdas = np.array([0.1, 0.2, 0.3])
        cs = np.array([dp.dp_opt(a_s, lamb, 0.0) for lamb in lambdas])
        self.assertTrue(np.allclose(cs, np.zeros((3, 6))))
        cs2 = dp.dp_opt(a_s, lambdas, 0.0)
        self.assertTrue(np.allclose(cs, cs2))

    def test_2(self):
        a_s = np.array([-1, -1, -1, 1, 1, 1])
        lambdas = np.array([0.0, 0.5, 1, 2.999, 3.001])
        cs = np.array([dp.dp_opt(a_s, lamb, 0.0) for lamb in lambdas])
        self.assertTrue(
            np.allclose(
                cs,
                np.array(
                    [
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        )
        cs2 = dp.dp_opt(a_s, lambdas, 0.0)
        self.assertTrue(np.allclose(cs, cs2))
