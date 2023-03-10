import argparse

import numpy as np
import vulkpy as vk
from vulkpy.util import enable_debug

def main():
    gpu = vk.GPU()

    shape = (100,)
    a = vk.Array(gpu, data=np.full(shape, 3))
    b = vk.Array(gpu, data=np.full(shape, 5))

    c = a + b
    c.wait()
    print(c)

    d = c - a
    e = d - b
    e.wait()
    print(e)

    e += a
    e.wait()
    print(e)

    f = e + 5
    f.wait()
    print(f)

    f /= 4
    f.wait()
    print(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser("00-arithmetic.py")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    if args.debug:
        enable_debug()

    main()
