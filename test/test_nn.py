import unittest

import numpy as np

import vulkpy as vk
from vulkpy import nn


class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_relu_forward(self):
        relu = nn.ReLU()

        x = vk.Array(self.gpu, data=[[-0.2, 0.0, 0.2]])
        y = relu(x)

        np.testing.assert_allclose(y, [[0.0, 0.0, 0.2]])


if __name__ == "__main__":
    unittest.main()
