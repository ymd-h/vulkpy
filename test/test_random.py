import unittest

import numpy as np
import vulkpy as vk


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

    @unittest.skip("")
    def test_higher_dimension(self):
        rng = vk.random.Xoshiro128pp(self.gpu)
        a = rng.random(shape=(5, 5, 5))

        a.wait()
        print(a)
        np.testing.assert_allclose(np.asarray(a).shape, (5, 5, 5))
        self.assertTrue((0 <= np.asarray(a)).all())
        self.assertTrue((np.asarray(a) < 1.0).all())

if __name__ == "__main__":
    unittest.main()
