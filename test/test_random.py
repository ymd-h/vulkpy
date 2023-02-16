import unittest

import numpy as np
import vulkpy as vk
from vulkpy.util import enable_debug


class TestRandom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_random(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = rng.random(shape=(3,))

        a.wait()
        np.testing.assert_allclose(np.asarray(a).shape, (3,))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

    def test_random_seed(self):
        rng1 = vk.random.Xoshiro128pp(self.gpu, seed=0)
        a = rng1.random(shape=(5,))

        rng2 = vk.random.Xoshiro128pp(self.gpu, seed=0)
        b = rng2.random(shape=(5,))

        a.wait()
        b.wait()
        np.testing.assert_allclose(a, b)
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())
        self.assertTrue((0 <= np.asarray(b)).all())
        self.assertTrue((np.asarray(b) < 1.0).all())

    def test_middle(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = rng.random(shape=(17,))

        a.wait()
        np.testing.assert_allclose(np.asarray(a).shape, (17,))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

    def test_larger(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = rng.random(shape=(65,))

        a.wait()
        np.testing.assert_allclose(np.asarray(a).shape, (65,))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

    def test_higher_dimension(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = rng.random(shape=(5, 5, 5))

        a.wait()
        np.testing.assert_allclose(np.asarray(a).shape, (5, 5, 5))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

    def test_buffer(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = vk.Array(self.gpu, shape=(5,))
        a = rng.random(buffer=a)
        a.wait()
        np.testing.assert_allclose(np.asarray(a).shape, (5,))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

    def test_normal(self):
        rng1 = vk.random.Xoshiro128pp(self.gpu, seed=0)
        rng2 = vk.random.Xoshiro128pp(self.gpu, seed=0)

        a1 = rng1.normal(shape=(10,))
        a2 = rng2.normal(shape=(10,), mean=5, stddev=3)

        np.testing.assert_allclose((a2 - 5) / a1, np.full((10,), 3), rtol=1e-6)

if __name__ == "__main__":
    enable_debug(api_dump=False)
    unittest.main()
