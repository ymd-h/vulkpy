import unittest

import numpy as np

import vulkpy as vk
from vulkpy import nn, random


class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_relu_forward(self):
        relu = nn.ReLU()

        x = vk.Array(self.gpu, data=[[-0.2, 0.0, 0.2]])
        y = relu(x)

        np.testing.assert_allclose(y, [[0.0, 0.0, 0.2]])

    def test_relu_backward(self):
        relu = nn.ReLU()

        x = vk.Array(self.gpu, data=[[-0.2, 0.0, 0.2]])
        y = relu(x)

        dy = vk.Array(self.gpu, data=[[0.7, 0.8, 0.9]])
        dx = relu.backward(dy)

        np.testing.assert_allclose(dx, [[0.0, 0.0, 0.9]])

    def test_he(self):
        seed = 645
        shape = (10,)

        he = nn.HeNormal(self.gpu, input_dim=2, seed=seed)
        self.assertEqual(he.stddev, 1.0)

        rng = random.Xoshiro128pp(self.gpu, seed=seed)

        np.testing.assert_allclose(he(self.gpu, shape), rng.normal(shape=shape))

if __name__ == "__main__":
    unittest.main()
