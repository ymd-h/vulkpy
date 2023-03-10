"""
Example 02: Neural Network for Classifying of Iris
==================================================

Classify 3-class Iris with Sequential Neural Network.
The hidden layers have units of 128 and 128, respectively.

For options, see `python 02-nn.py -h`.

Notes
-----
This example requires scikit-learn (`pip install scikit-learn`)
"""
import argparse
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import vulkpy as vk
from vulkpy.util import enable_debug
from vulkpy import nn


def example02(nepoch, batch_size, opt, lr, l1, l2, *, debug = False):
    if debug:
        enable_debug(api_dump=False)

    gpu = vk.GPU()
    rng = np.random.default_rng()

    train_x, test_x, train_y, test_y = train_test_split(*load_iris(return_X_y=True),
                                                        random_state = 777,
                                                        test_size = 0.2)

    # Convert to one_hot vector
    train_y = np.identity(3)[train_y]
    test_y = np.identity(3)[test_y]

    print(f"train_x.shape: {train_x.shape}, test_x.shape: {test_x.shape}")
    print(f"train_y.shape: {train_y.shape}, test_y.shape: {test_y.shape}")
    assert ((train_x.shape[0] == train_y.shape[0]) and
            ( test_x.shape[0] ==  test_y.shape[0]))
    assert train_x.shape[1] == test_x.shape[1] == 4
    assert train_y.shape[1] == test_y.shape[1] == 3

    opt = {
        "adam": lambda lr: nn.Adam(gpu, lr=lr),
        "sgd": lambda lr: nn.SGD(lr)
    }[opt](lr)

    R = None
    if (l1 is not None) and (l2 is not None):
        R = nn.Elastic(l1, l2)
    elif (l1 is not None):
        R = nn.Lasso(l1)
    elif (l2 is not None):
        R = nn.Ridge(l2)

    # Sequential Model: 4 -> 128 -> 128 -> 3
    net = nn.Sequence(
        [
            nn.Dense(gpu, 4, 128, w_opt=opt, b_opt=opt, w_reg=R, b_reg=R),
            nn.ReLU(),
            nn.Dense(gpu, 128, 128, w_opt=opt, b_opt=opt, w_reg=R, b_reg=R),
            nn.ReLU(),
            nn.Dense(gpu, 128, 3, w_opt=opt, b_opt=opt, w_reg=R, b_reg=R),
            nn.Softmax(),
         ],
        nn.CrossEntropyLoss(reduce="sum")
    )
    idx = np.arange(train_x.shape[0])

    X = vk.Array(gpu, data=test_x)
    Y = vk.Array(gpu, data=test_y)

    train_loss = vk.Array(gpu, shape=(1,))
    for e in range(nepoch):
        t = time.perf_counter()

        rng.shuffle(idx) # TODO: Implement GPU shuffle()
        train_loss[:] = 0
        for _idx in idx[::batch_size]:
            bidx = idx[_idx:_idx+batch_size]

            x = vk.Array(gpu, data=train_x[bidx])
            y = vk.Array(gpu, data=train_y[bidx])

            _, loss = net.train(x, y)
            train_loss += loss

        train_loss /= idx.shape[0]

        pred_y, eval_loss = net.predict(X, Y)
        pred_class = np.argmax(pred_y, axis=1) # TODO: Implement GPU argmax()
        accuracy = (np.identity(3)[pred_class] * test_y).sum(axis=1).mean()

        eval_loss /= idx.shape[0]

        dt = time.perf_counter() - t
        print(f"Epoch: {e:3d}, " +
              f"Train Loss: {train_loss[0]:.6f}, " +
              f"Eval Loss: {float(eval_loss.array):.6f}, " +
              f"Eval Acc: {accuracy:.6f} " +
              f"Elapsed: {dt:.6f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser("example02")
    p.add_argument("--nepoch", type=int, default=100, help="# of epoch")
    p.add_argument("--batch-size", type=int, default=32, help="size of batch")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--l1", type=float, help="L1 regularization", default=None)
    p.add_argument("--l2", type=float, help="L2 regularization", default=None)
    p = p.parse_args()

    example02(nepoch=p.nepoch,
              batch_size=p.batch_size,
              opt=p.optimizer,
              lr=p.learning_rate,
              l1=p.l1,
              l2=p.l2,
              debug=p.debug)
