import unittest

import numpy as np
import vulkpy as vk
from vulkpy.util import enable_debug


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

    def test_cascade_add(self):
        a = vk.Array(self.gpu, data=[5, 5, 5])
        b = vk.Array(self.gpu, data=[1, 1, 1])
        c = vk.Array(self.gpu, data=[4, 4, 4])

        d = a + b + c
        d.wait()

        np.testing.assert_allclose(d, np.asarray([10, 10, 10]))


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


    def test_incompatible(self):
        a = vk.Array(self.gpu, data=[1, 1])
        b = vk.Array(self.gpu, data=[1, 1, 1])

        with self.assertRaises(ValueError):
            c = a + b

    def test_higher_dimension(self):
        a = vk.Array(self.gpu, data=[[1, 1], [1, 1]])
        b = vk.Array(self.gpu, data=[[2, 2], [2, 2]])

        c = a + b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([[3, 3], [3, 3]]))

    def test_matmul(self):
        a = vk.Array(self.gpu, data=[[1, 2],
                                     [3, 4]])
        b = vk.Array(self.gpu, data=[[1, 2],
                                     [3, 4]])

        c = a @ b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([[ 7, 10],
                                                  [15, 22]]))

    def test_matvec(self):
        a = vk.Array(self.gpu, data=[[1, 2],
                                     [3, 4]])
        b = vk.Array(self.gpu, data=[1, 3])

        c = a @ b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([7, 15]))

    def test_vecmat(self):
        a = vk.Array(self.gpu, data=[1, 2])
        b = vk.Array(self.gpu, data=[[1, 2],
                                     [3, 4]])

        c = a @ b
        c.wait()

        np.testing.assert_allclose(c, np.asarray([7, 10]))

    def test_incompatible_matmul(self):
        a = vk.Array(self.gpu, data=[[1, 2, 3], [4, 5, 6]])
        b = vk.Array(self.gpu, data=[[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            c = a @ b

    def test_reshape(self):
        a = vk.Array(self.gpu, data=[1, 2, 3, 4])
        a.reshape((2, 2))

        np.testing.assert_allclose(a, np.asarray([[1, 2], [3, 4]]))

    def test_incompatible_reshape(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        with self.assertRaises(ValueError):
            a.reshape((2, 2))

    def test_max(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = vk.Array(self.gpu, data=[3, 2, 1])

        c = a.max(b)
        c.wait()

        np.testing.assert_allclose(c, np.asarray([3, 2, 3]))

    def test_imax(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = vk.Array(self.gpu, data=[3, 2, 1])

        a.max(b, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([3, 2, 3]))

    def test_min(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = vk.Array(self.gpu, data=[3, 2, 1])

        c = a.min(b)
        c.wait()

        np.testing.assert_allclose(c, np.asarray([1, 2, 1]))

    def test_imin(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = vk.Array(self.gpu, data=[3, 2, 1])

        a.min(b, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([1, 2, 1]))

    def test_max_scalar(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = a.max(1.5)
        b.wait()

        np.testing.assert_allclose(b, np.asarray([1.5, 2, 3]))

    def test_imax_scalar(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        a.max(1.5, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([1.5, 2, 3]))

    def test_min_scalar(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = a.min(1.5)
        b.wait()

        np.testing.assert_allclose(b, np.asarray([1, 1.5, 1.5]))

    def test_imin_scalar(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        a.min(1.5, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([1, 1.5, 1.5]))

    def test_abs(self):
        a = vk.Array(self.gpu, data=[-1, 1, 3])
        b = a.abs()
        b.wait()

        np.testing.assert_allclose(b, np.asarray([1, 1, 3]))

    def test_iabs(self):
        a = vk.Array(self.gpu, data=np.asarray([-1, 1, 3]))
        a.abs(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([1, 1, 3]))

    def test_sign(self):
        a = vk.Array(self.gpu, data=[-1, 1, 3])
        b = a.sign()
        b.wait()

        np.testing.assert_allclose(b, np.asarray([-1, 1, 1]))

    def test_isign(self):
        a = vk.Array(self.gpu, data=[-1, 1, 3])
        a.sign(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.asarray([-1, 1, 1]))

    def test_sin(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.sin()
        b.wait()

        np.testing.assert_allclose(b, np.sin(x), rtol=1e-5)

    def test_isin(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.sin(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.sin(x), rtol=1e-5)

    def test_cos(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.cos()
        b.wait()

        np.testing.assert_allclose(b, np.cos(x), rtol=1e-5)

    def test_icos(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.cos(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.cos(x), rtol=1e-5)

    def test_tan(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.tan()
        b.wait()

        np.testing.assert_allclose(b, np.tan(x), rtol=1e-5)

    def test_itan(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.tan(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.tan(x), rtol=1e-5)

    def test_sinh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.sinh()
        b.wait()

        np.testing.assert_allclose(b, np.sinh(x), rtol=1e-5)

    def test_isinh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.sinh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.sinh(x), rtol=1e-5)

    def test_cosh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.cosh()
        b.wait()

        np.testing.assert_allclose(b, np.cosh(x), rtol=1e-5)

    def test_icosh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.cosh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.cosh(x), rtol=1e-5)

    def test_tanh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        b = a.tanh()
        b.wait()

        np.testing.assert_allclose(b, np.tanh(x), rtol=1e-5)

    def test_itanh(self):
        x = np.asarray([1, 3, 5])
        a = vk.Array(self.gpu, data=x)
        a.tanh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.tanh(x), rtol=1e-5)

    def test_asin(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        b = a.asin()
        b.wait()

        np.testing.assert_allclose(b, np.arcsin(x), rtol=1e-3)

    def test_iasin(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        a.asin(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arcsin(x), rtol=1e-3)

    def test_acos(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        b = a.acos()
        b.wait()

        np.testing.assert_allclose(b, np.arccos(x), rtol=1e-3)

    def test_iacos(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        a.acos(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arccos(x), rtol=1e-3)

    def test_atan(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        b = a.atan()
        b.wait()

        np.testing.assert_allclose(b, np.arctan(x), rtol=1e-5)

    def test_iatan(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        a.atan(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arctan(x), rtol=1e-5)

    def test_iasinh(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        a.asinh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arcsinh(x), rtol=1e-5)

    def test_acosh(self):
        x = np.asarray([1.5, 1.3, 1.2])
        a = vk.Array(self.gpu, data=x)
        b = a.acosh()
        b.wait()

        np.testing.assert_allclose(b, np.arccosh(x), rtol=1e-5)

    def test_iacosh(self):
        x = np.asarray([1.5, 1.3, 1.2])
        a = vk.Array(self.gpu, data=x)
        a.acosh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arccosh(x), rtol=1e-5)

    def test_atanh(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        b = a.atanh()
        b.wait()

        np.testing.assert_allclose(b, np.arctanh(x), rtol=1e-5)

    def test_iatanh(self):
        x = np.asarray([0.5, 0.3, -0.2])
        a = vk.Array(self.gpu, data=x)
        a.atanh(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.arctanh(x), rtol=1e-5)

    def test_exp(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.exp()
        b.wait()

        np.testing.assert_allclose(b, np.exp(x))

    def test_iexp(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.exp(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.exp(x))

    def test_log(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.log()
        b.wait()

        np.testing.assert_allclose(b, np.log(x))

    def test_ilog(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.log(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.log(x))

    def test_exp2(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.exp2()
        b.wait()

        np.testing.assert_allclose(b, np.exp2(x))

    def test_iexp2(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.exp2(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.exp2(x))

    def test_log2(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.log2()
        b.wait()

        np.testing.assert_allclose(b, np.log2(x))

    def test_ilog2(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.log2(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.log2(x))

    def test_sqrt(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.sqrt()
        b.wait()

        np.testing.assert_allclose(b, np.sqrt(x))

    def test_isqrt(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.sqrt(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.sqrt(x))

    def test_invsqrt(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        b = a.invsqrt()
        b.wait()

        np.testing.assert_allclose(b, np.sqrt(1/x))

    def test_iinvsqrt(self):
        x = np.asarray([1, 2, 3])
        a = vk.Array(self.gpu, data=x)
        a.invsqrt(inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.sqrt(1/x))

    def test_pow(self):
        x = np.asarray([1, 2, 3])
        y = np.asarray([1.1, 2.2, 1.4])

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=y)

        c = a ** b
        c.wait()

        np.testing.assert_allclose(c, x ** y)

    def test_ipow(self):
        x = np.asarray([1, 2, 3])
        y = np.asarray([1.1, 2.2, 1.4])

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=y)

        a **= b
        a.wait()

        np.testing.assert_allclose(a, x ** y)

    def test_pow_scalar(self):
        x = np.asarray([1, 2, 3])
        y = 2.7

        a = vk.Array(self.gpu, data=x)

        c = a ** y
        c.wait()

        np.testing.assert_allclose(c, x ** y)

    def test_ipow_scalar(self):
        x = np.asarray([1, 2, 3])
        y = 2.7

        a = vk.Array(self.gpu, data=x)

        a **= y
        a.wait()

        np.testing.assert_allclose(a, x ** y)

    def test_rpow_scalar(self):
        x = 1.3
        y = np.asarray([1.1, 2.2, 1.4])

        b = vk.Array(self.gpu, data=y)

        c = x ** b
        c.wait()

        np.testing.assert_allclose(c, x ** y)

    def test_clamp(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = np.asarray([3, 3, 4])
        _max = np.asarray([6, 6, 7])

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=_min)
        c = vk.Array(self.gpu, data=_max)

        d = a.clamp(b, c)
        d.wait()

        np.testing.assert_allclose(d, np.clip(x, _min, _max))

    def test_iclamp(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = np.asarray([3, 3, 4])
        _max = np.asarray([6, 6, 7])

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=_min)
        c = vk.Array(self.gpu, data=_max)

        a.clamp(b, c, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.clip(x, _min, _max))

    def test_clamp_sv(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = 3
        _max = np.asarray([6, 6, 7])

        a = vk.Array(self.gpu, data=x)
        c = vk.Array(self.gpu, data=_max)

        d = a.clamp(_min, c)
        d.wait()

        np.testing.assert_allclose(d, np.clip(x, _min, _max))

    def test_iclamp_sv(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = 3
        _max = np.asarray([6, 6, 7])

        a = vk.Array(self.gpu, data=x)
        c = vk.Array(self.gpu, data=_max)

        a.clamp(_min, c, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.clip(x, _min, _max))

    def test_clamp_vs(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = np.asarray([3, 3, 4])
        _max = 7

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=_min)

        d = a.clamp(b, _max)
        d.wait()

        np.testing.assert_allclose(d, np.clip(x, _min, _max))

    def test_iclamp_vs(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = np.asarray([3, 3, 4])
        _max = 7

        a = vk.Array(self.gpu, data=x)
        b = vk.Array(self.gpu, data=_min)

        a.clamp(b, _max, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.clip(x, _min, _max))

    def test_clamp_ss(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = 3
        _max = 7

        a = vk.Array(self.gpu, data=x)

        d = a.clamp(_min, _max)
        d.wait()

        np.testing.assert_allclose(d, np.clip(x, _min, _max))

    def test_iclamp_ss(self):
        x = np.asarray([1.2, 3.5, 10])
        _min = 3
        _max = 7

        a = vk.Array(self.gpu, data=x)

        a.clamp(_min, _max, inplace=True)
        a.wait()

        np.testing.assert_allclose(a, np.clip(x, _min, _max))

    def test_sum(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])
        b = a.sum()
        b.wait()

        np.testing.assert_allclose(b, (6,))

    def test_sum_large(self):
        a = vk.Array(self.gpu, data=np.ones(shape=(65,)))
        b = a.sum()
        b.wait()

        np.testing.assert_allclose(b, (65,))

    def test_sum_axis(self):
        a = vk.Array(self.gpu, data=[1, 2, 3])

        b = a.sum(axis=0)
        b.wait()

        np.testing.assert_allclose(b, (6, ))

    def test_sum_axis_multi(self):
        x = np.asarray([[1, 2], [3, 4]])
        a = vk.Array(self.gpu, data=x)

        b = a.sum(axis=1)
        b.wait()

        np.testing.assert_allclose(b, x.sum(axis=1))

    def test_sum_axis_multi_axis(self):
        x = np.ones(shape=(2,3,4,2))
        a = vk.Array(self.gpu, data=x)

        b = a.sum(axis=(1, 2))
        b.wait()

        np.testing.assert_allclose(b, x.sum(axis=(1, 2)))

    def test_sum_axis_multi_axis_large(self):
        x = np.ones(shape=(2,3,4,2,2,4,3))
        a = vk.Array(self.gpu, data=x)

        b = a.sum(axis=(1, 2, 5))
        b.wait()

        np.testing.assert_allclose(b, x.sum(axis=(1, 2, 5)))

if __name__ == "__main__":
    enable_debug(api_dump=False)
    unittest.main()
