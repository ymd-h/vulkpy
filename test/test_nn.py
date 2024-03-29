import unittest

import numpy as np

import vulkpy as vk
from vulkpy import nn, random


class TestInitializers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_constant(self):
        const = nn.Constant(0.0)
        np.testing.assert_allclose(const(self.gpu, (3,1)), [[0.0], [0.0], [0.0]])

    def test_he(self):
        seed = 645
        shape = (10,)

        he = nn.HeNormal(self.gpu, input_dim=2, seed=seed)
        self.assertEqual(he.stddev, 1.0)

        rng = random.Xoshiro128pp(self.gpu, seed=seed)

        np.testing.assert_allclose(he(self.gpu, shape), rng.normal(shape=shape))


class TestOptimizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_sgd(self):
        sgd = nn.SGD(lr=0.01)

        grad = vk.Array(self.gpu, data=[1, 2, 3])
        state = sgd.init_state(grad.shape)

        diff = state.grad2diff(grad)
        np.testing.assert_allclose(diff, grad * (-0.01))

    def test_adam(self):
        adam = nn.Adam(self.gpu)

        grad = vk.Array(self.gpu, data=[1, 2, 3])
        state = adam.init_state(grad.shape)
        self.assertEqual(state.beta1t, 1.0)
        self.assertEqual(state.beta2t, 1.0)

        diff = state.grad2diff(grad)

        self.assertEqual(state.beta1t, adam.beta1)
        self.assertEqual(state.beta2t, adam.beta2)

    def test_adagrad(self):
        adagrad = nn.AdaGrad(self.gpu)

        grad = vk.Array(self.gpu, data=[1, 2, 3])
        state = adagrad.init_state(grad.shape)

        diff = state.grad2diff(grad)

class TestLayers(unittest.TestCase):
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

    def test_sigmoid_forward(self):
        sigmoid = nn.Sigmoid()

        d = np.asarray([[-100, -0.1, 0, 10,  100]])
        x = vk.Array(self.gpu, data=d)

        y = sigmoid(x)

        np.testing.assert_allclose(y, 1/(1+np.exp(-d)), rtol=1e-7, atol=1e-7)

    def test_sigmoid_backward(self):
        sigmoid = nn.Sigmoid()

        _x = np.asarray([[-100, -0.1, 0, 10,  100]])
        x = vk.Array(self.gpu, data=_x)
        y = sigmoid(x)

        _dy = np.asarray([[0.1, 0.2, 0.3, 0.5, 0.7]])
        dy = vk.Array(self.gpu, data=_dy)

        dx = sigmoid.backward(dy)
        np.testing.assert_allclose(dx, dy * y * (1 - y))

    def test_softmax(self):
        softmax = nn.Softmax()

        x = vk.Array(self.gpu, data=[[1.0, 1.0]])
        y = softmax(x)

        np.testing.assert_allclose(y, [[0.5, 0.5]])

    def test_softmax_skew(self):
        softmax = nn.Softmax()

        x = vk.Array(self.gpu, data=[[100.0, 0.0]])
        y = softmax(x)

        np.testing.assert_allclose(y, [[1.0, 0]])

    def test_softmax_forward(self):
        softmax = nn.Softmax()

        _x = np.asarray([[-100, -0.1, 0, 10, 100]])
        x = vk.Array(self.gpu, data=_x)

        y = softmax(x)

        exp_x = np.exp(_x - _x.max(axis=1))
        np.testing.assert_allclose(y, exp_x / exp_x.sum(axis=1, keepdims=True),
                                   rtol=1e-7, atol=1e-7)

    def test_softmax_backward(self):
        softmax = nn.Softmax()

        _x = np.asarray([[-100, -0.1, 0, 10, 100]])
        x = vk.Array(self.gpu, data=_x)

        y = softmax(x)

        _dy = np.asarray([[0.1, 0.2, 0.3, 0.5, 0.7]])
        dy = vk.Array(self.gpu, data=_dy)

        dx = softmax.backward(dy)

        np.testing.assert_allclose(dx, dy * y * (1 - y))

    def test_dense_zero(self):
        dense = nn.Dense(self.gpu, 2, 2, w_init=nn.Constant(0.0))

        x = vk.Array(self.gpu, data=[[1, 2], [3, 4]])
        y = dense(x)

        np.testing.assert_allclose(y, [[0, 0], [0, 0]])

    def test_dense_bias(self):
        dense = nn.Dense(self.gpu, 2, 2,
                         w_init=nn.Constant(0.0),
                         b_init=nn.Constant(1.0))

        x = vk.Array(self.gpu, data=[[1,2], [3,4]])
        y = dense(x)

        np.testing.assert_allclose(y, [[1, 1], [1, 1]])

    def test_dense(self):
        dense = nn.Dense(self.gpu, 2, 2)

        x = vk.Array(self.gpu, data=[[2, 3], [2, 3]])
        y = dense(x)

        np.testing.assert_allclose(y[0,:], y[1,:])

    def test_dense_backward(self):
        dense = nn.Dense(self.gpu, 2, 2)
        np.testing.assert_allclose(dense.w.grad, [[0, 0], [0, 0]])
        np.testing.assert_allclose(dense.b.grad, [0, 0])

        x = vk.Array(self.gpu, data=[[1, 2], [3, 4]])
        y = dense(x)

        dy = vk.Array(self.gpu, data=[[4, 2], [1, 3]])
        dx = dense.backward(dy)

        np.testing.assert_allclose(dense.w.grad, [[7, 12], [11, 16]])
        np.testing.assert_allclose(dense.b.grad, [5, 5])

        _w = dense.w.value
        np.testing.assert_allclose(dx,
                                   [[_w[0,0] * dy[0,0] + _w[1,0] * dy[0,1],
                                     _w[0,1] * dy[0,0] + _w[1,1] * dy[0,1]],
                                    [_w[0,0] * dy[1,0] + _w[1,0] * dy[1,1],
                                     _w[0,1] * dy[1,0] + _w[1,1] * dy[1,1]]])

class TestLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = vk.GPU()

    def test_cross_entropy(self):
        loss = nn.CrossEntropyLoss()

        x = vk.Array(self.gpu, data=[[1.0, 0.0]])
        y = vk.Array(self.gpu, data=[[1.0, 0.0]])

        L = loss(x, y)
        np.testing.assert_allclose(L, [0.0])

    def test_cross_entropy_equal(self):
        loss = nn.CrossEntropyLoss()

        x = vk.Array(self.gpu, data=[[0.5, 0.5]])
        y = vk.Array(self.gpu, data=[[0.5, 0.5]])

        L = loss(x, y)
        np.testing.assert_allclose(L, [0.6931472])

    def test_cross_entropy_default(self):
        loss = nn.CrossEntropyLoss()

        _x = np.asarray([[0.7, 0.3], [0.2, 0.8], [1.0, 0.0]])
        _y = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        x = vk.Array(self.gpu, data=_x)
        y = vk.Array(self.gpu, data=_y)

        _L = np.sum(-_y * np.log(_x + 1e-8), axis=1)
        L = loss(x, y)
        np.testing.assert_allclose(L, _L.mean(), atol=1e-7, rtol=1e-7)

        dx = loss.grad()
        _dx = - _y / (_x + 1e-8)
        np.testing.assert_allclose(dx, _dx / _dx.shape[0])

    def test_cross_entropy_mean(self):
        loss = nn.CrossEntropyLoss(reduce="mean")

        _x = np.asarray([[0.7, 0.3], [0.2, 0.8], [1.0, 0.0]])
        _y = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        x = vk.Array(self.gpu, data=_x)
        y = vk.Array(self.gpu, data=_y)

        _L = np.sum(-_y * np.log(_x + 1e-8), axis=1)
        L = loss(x, y)
        np.testing.assert_allclose(L, _L.mean(), atol=1e-7, rtol=1e-7)

        dx = loss.grad()
        _dx = - _y / (_x + 1e-8)
        np.testing.assert_allclose(dx, _dx / _dx.shape[0])

    def test_cross_entropy_sum(self):
        loss = nn.CrossEntropyLoss(reduce="sum")

        _x = np.asarray([[0.7, 0.3], [0.2, 0.8], [1.0, 0.0]])
        _y = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        x = vk.Array(self.gpu, data=_x)
        y = vk.Array(self.gpu, data=_y)

        _L = np.sum(-_y * np.log(_x + 1e-8), axis=1)
        L = loss(x, y)
        np.testing.assert_allclose(L, _L.sum(), atol=1e-7, rtol=1e-7)

        dx = loss.grad()
        _dx = - _y / (_x + 1e-8)
        np.testing.assert_allclose(dx, _dx)

    def test_softmax_crossentropy(self):
        sce = nn.SoftmaxCrossEntropyLoss()

        _x = np.asarray([[100.0, 0.0]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1.0, 0.0]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)
        np.testing.assert_allclose(L, [0.0])

    def test_softmax_crossentropy_forward_default(self):
        sce = nn.SoftmaxCrossEntropyLoss()

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = (-_y * np.log(exp_x / exp_x.sum(axis=1, keepdims=True))).sum(axis=1)
        np.testing.assert_allclose(L, _L.mean(axis=0), atol=1e-7, rtol=1e-7)

    def test_softmax_crossentropy_forward_mean(self):
        sce = nn.SoftmaxCrossEntropyLoss(reduce="mean")

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = (-_y * np.log(exp_x / exp_x.sum(axis=1, keepdims=True))).sum(axis=1)
        np.testing.assert_allclose(L, _L.mean(axis=0), atol=1e-7, rtol=1e-7)

    def test_softmax_crossentropy_forward_sum(self):
        sce = nn.SoftmaxCrossEntropyLoss(reduce="sum")

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = (-_y * np.log(exp_x / exp_x.sum(axis=1, keepdims=True))).sum(axis=1)
        np.testing.assert_allclose(L, _L.sum(axis=0), atol=1e-7, rtol=1e-7)

    def test_softmax_crossentropy_backward_default(self):
        sce = nn.SoftmaxCrossEntropyLoss()

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        dx = sce.grad()

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = exp_x / exp_x.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(dx, (_L - _y) / _y.shape[0], atol=1e-7, rtol=1e-7)

    def test_softmax_crossentropy_backward_mean(self):
        sce = nn.SoftmaxCrossEntropyLoss(reduce="mean")

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        dx = sce.grad()

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = exp_x / exp_x.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(dx, (_L - _y) / _y.shape[0], atol=1e-7, rtol=1e-7)

    def test_softmax_crossentropy_backward_sum(self):
        sce = nn.SoftmaxCrossEntropyLoss(reduce="sum")

        _x = np.asarray([[-1, 0], [10, 15]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[1, 0], [0, 1]])
        y = vk.Array(self.gpu, data=_y)

        L = sce(x, y)

        dx = sce.grad()

        exp_x = np.exp(_x - _x.max(axis=1, keepdims=True))
        _L = exp_x / exp_x.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(dx, _L - _y, atol=1e-7, rtol=1e-7)

    def test_mse_loss_default(self):
        mse = nn.MSELoss()

        _x = np.asarray([[4, 2], [1, 1.5]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[3, 2.2], [0.7, 1.5]])
        y = vk.Array(self.gpu, data=_y)

        L = mse(x, y)
        dx = mse.grad()

        np.testing.assert_allclose(L, np.square(_y - _x).sum(axis=1).mean(axis=0),
                                   atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(dx, (_x - _y),
                                   atol=1e-7, rtol=1e-7)

    def test_mse_loss_mean(self):
        mse = nn.MSELoss(reduce="mean")

        _x = np.asarray([[4, 2], [1, 1.5]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[3, 2.2], [0.7, 1.5]])
        y = vk.Array(self.gpu, data=_y)

        L = mse(x, y)
        dx = mse.grad()

        np.testing.assert_allclose(L, np.square(_y - _x).sum(axis=1).mean(axis=0),
                                   atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(dx, (_x - _y),
                                   atol=1e-7, rtol=1e-7)

    def test_mse_loss_sum(self):
        mse = nn.MSELoss(reduce="sum")

        _x = np.asarray([[4, 2], [1, 1.5]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[3, 2.2], [0.7, 1.5]])
        y = vk.Array(self.gpu, data=_y)

        L = mse(x, y)
        dx = mse.grad()

        np.testing.assert_allclose(L, np.square(_y - _x).sum(axis=1).sum(axis=0),
                                   atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(dx, 2 * (_x - _y),
                                   atol=1e-7, rtol=1e-7)

    def test_huber_loss_default(self):
        huber = nn.HuberLoss()

        _x = np.asarray([[1.0, 2.2], [-3.0, 0.7]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[10, 3.0], [-5, 0.5]])
        y = vk.Array(self.gpu, data=_y)

        L = huber(x, y)
        dx = huber.grad()

        np.testing.assert_allclose(L, [2.92])
        np.testing.assert_allclose(dx, [[-0.5, -0.4], [0.5, 0.1]])

    def test_huber_loss_mean(self):
        huber = nn.HuberLoss(reduce="mean")

        _x = np.asarray([[1.0, 2.2], [-3.0, 0.7]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[10, 3.0], [-5, 0.5]])
        y = vk.Array(self.gpu, data=_y)

        L = huber(x, y)
        dx = huber.grad()

        np.testing.assert_allclose(L, [2.92])
        np.testing.assert_allclose(dx, [[-0.5, -0.4], [0.5, 0.1]])

    def test_huber_loss_sum(self):
        huber = nn.HuberLoss(reduce="sum")

        _x = np.asarray([[1.0, 2.2], [-3.0, 0.7]])
        x = vk.Array(self.gpu, data=_x)

        _y = np.asarray([[10, 3.0], [-5, 0.5]])
        y = vk.Array(self.gpu, data=_y)

        L = huber(x, y)
        dx = huber.grad()

        np.testing.assert_allclose(L, [5.84])
        np.testing.assert_allclose(dx, [[-1.0, -0.8], [1.0, 0.2]])

class TestRegularizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vulkpy.nn.parameters import Parameter
        cls.gpu = vk.GPU()
        cls.P = Parameter

    def test_ridge_zero(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(0.0))
        R = nn.Ridge(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(0.0))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((0.0, )))

    def test_ridge(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(3.5))
        R = nn.Ridge(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(3.5 ** 2))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((2 * 3.5,)))

    def test_ridge_negative(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(-3.5))
        R = nn.Ridge(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray((-3.5) ** 2))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((2 * -3.5,)))

    def test_lasso_zero(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(0.0))
        R = nn.Lasso(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(0.0))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((0.0, )))

    def test_lasso(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(3.5))
        R = nn.Lasso(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(3.5))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((1.0,)))

    def test_lasso_negative(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(-3.5))
        R = nn.Lasso(1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(3.5))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((-1.0,)))

    def test_elastic_zero(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(0.0))
        R = nn.Elastic(1.0, 1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(0.0))
        np.testing.assert_allclose(R.grad(p.value), np.asarray(0.0,))

    def test_elastic(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(3.5))
        R = nn.Elastic(1.0, 1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(3.5 ** 2 + 3.5))

        np.testing.assert_allclose(R.grad(p.value), np.asarray((2 * 3.5 + 1.0,)))

    def test_elastic_negative(self):
        p = self.P(self.gpu, (1,), initializer=nn.Constant(-3.5))
        R = nn.Elastic(1.0, 1.0)

        np.testing.assert_allclose(R.loss(p.value), np.asarray(3.5 ** 2 + 3.5))
        np.testing.assert_allclose(R.grad(p.value), np.asarray((2 * -3.5 - 1.0,)))

if __name__ == "__main__":
    unittest.main()
