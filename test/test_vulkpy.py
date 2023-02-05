import unittest

import numpy as np
import vulkpy as vk


class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_add(self):
        a = vk.Array(self.gpu, data=[5, 5, 5])
        b = vk.Array(self.gpu, data=[1, 1, 1])
        c = a + b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([6, 6, 6]))

    def test_sub(self):
        a = vk.Array(self.gpu, data=[4, 4, 4])
        b = vk.Array(self.gpu, data=[2, 2, 2])
        c = a - b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([2, 2, 2]))

    def test_mul(self):
        a = vk.Array(self.gpu, data=[2, 2, 2])
        b = vk.Array(self.gpu, data=[3, 3, 3])
        c = a * b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([6, 6, 6]))

    def test_div(self):
        a = vk.Array(self.gpu, data=[8, 8, 8])
        b = vk.Array(self.gpu, data=[2, 2, 2])
        c = a / b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([4, 4, 4]))

    def test_iadd(self):
        a = vk.Array(self.gpu, data=[2, 2, 2])
        b = vk.Array(self.gpu, data=[5, 5, 5])
        a += b
        a.wait()

        np.testing.assert_allclose(a, np.asarray([7, 7, 7]))

    def test_isub(self):
        a = vk.Array(self.gpu, data=[7, 7, 7])
        b = vk.Array(self.gpu, data=[3, 3, 3])
        a -= b
        a.wait()

        np.testing.assert_allclose(a, np.asarray([4, 4, 4]))

    def test_imul(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        b = vk.Array(self.gpu, data=[2, 2, 2])
        a *= b
        a.wait()

        np.testing.assert_allclose(a, np.asarray([6, 6, 6]))

    def test_idiv(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        b = vk.Array(self.gpu, data=[2, 2, 2])
        a /= b
        a.wait()

        np.testing.assert_allclose(a, np.asarray([1.5, 1.5, 1.5]))

    def test_add_scalar(self):
        a = vk.Array(self.gpu, data=[2, 2, 2])
        b = a * 5
        b.wait()

        np.testing.assert_allclose(b, np.asarray([10, 10, 10]))

    def test_sub_scalar(self):
        a = vk.Array(self.gpu, data=[5, 5, 5])
        b = a - 3
        b.wait()

        np.testing.assert_allclose(b, np.asarray([2, 2, 2]))

    def test_mul_scalar(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        b = a * 3
        b.wait()

        np.testing.assert_allclose(b, np.asarray([9, 9, 9]))

    def test_div_scalar(self):
        a = vk.Array(self.gpu, data=[5, 5, 5])
        b = a / 2
        b.wait()

        np.testing.assert_allclose(b, np.asarray([2.5, 2.5, 2.5]))

    def test_iadd_scalar(self):
        a = vk.Array(self.gpu, data=[2, 2, 2])
        a += 5
        a.wait()

        np.testing.assert_allclose(a, np.asarray([7, 7, 7]))

    def test_isub_scalar(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        a -= 1
        a.wait()

        np.testing.assert_allclose(a, np.asarray([2, 2, 2]))

    def test_imul_scalar(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        a *= 5
        a.wait()

        np.testing.assert_allclose(a, np.asarray([15, 15, 15]))

    def test_idiv_scalar(self):
        a = vk.Array(self.gpu, data=[8, 8, 8])
        a /= 4
        a.wait()

        np.testing.assert_allclose(a, np.asarray([2, 2, 2]))

    def test_radd_scalar(self):
        a = vk.Array(self.gpu, data=[5, 5, 5])
        b = 3 + a
        b.wait()

        np.testing.assert_allclose(b, np.asarray([8, 8, 8]))

    def test_rsub_scalar(self):
        a = vk.Array(self.gpu, data=[6, 6, 6])
        b = 4 - a
        b.wait()

        np.testing.assert_allclose(b, np.asarray([-2, -2, -2]))

    def test_rmul_scalar(self):
        a = vk.Array(self.gpu, data=[3, 3, 3])
        b = 7 * a
        b.wait()

        np.testing.assert_allclose(b, np.asarray([21, 21, 21]))

    def test_rdiv_scalar(self):
        a = vk.Array(self.gpu, data=[2, 2, 2])
        b = 8 / a
        b.wait()

        np.testing.assert_allclose(b, np.asarray([4, 4, 4]))

    def test_write(self):
        a = vk.Array(self.gpu, shape=(3,))
        a[:] = 10

        np.testing.assert_allclose(a, np.asarray([10, 10, 10]))

    def test_write_then_op(self):
        a = vk.Array(self.gpu, shape=(3,))
        b = vk.Array(self.gpu, shape=(3,))

        a[:] = 5
        b[:] = 7
        c = a + b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([12, 12, 12]))

    def test_chain_ops(self):
        a = vk.Array(self.gpu, data=[1, 1, 1])
        b = vk.Array(self.gpu, data=[2, 2, 2])

        c = a + b
        d = a + c
        d.wait()

        np.testing.assert_allclose(d, np.asarray([4, 4, 4]))

if __name__ == "__main__":
    unittest.main()
