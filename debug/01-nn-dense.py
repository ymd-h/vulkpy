import argparse

import numpy as np

import vulkpy as vk
from vulkpy.util import enable_debug


def debug01(epoch):
    gpu = vk.GPU()
    dense = vk.nn.Dense(gpu, 1, 1)
    mse = vk.nn.MSELoss()

    _x = np.arange(100).reshape((-1, 1)) / 50 - 1.0
    _y = _x ** 2

    x = vk.Array(gpu, data=_x)
    y = vk.Array(gpu, data=_y)

    for e in range(epoch):
        L = mse(dense(x), y)

        dense.zero_grad()
        dx = dense.backward(mse.grad())
        dense.update()

        print(f"Epoch: {e:4d}, Loss: {L:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("debug-01")
    p.add_argument("--api-dump", action="store_true")
    p.add_argument("--epoch", type=int, default=100)
    p = p.parse_args()

    enable_debug(api_dump=p.api_dump)
    debug01(p.epoch)
