import argparse

import numpy as np
import vulkpy as vk
from vulkpy.util import enable_debug

def main():
    gpu = vk.GPU()

    r = vk.random.Xoshiro128pp(gpu)

    # Sample from [0, 1) uniform distribution
    a = r.random(shape=(10,))
    print(a)

    # Sample from normal distribution
    b = r.normal(shape=(10,))
    print(b)


if __name__ == "__main__":
    p = argparse.ArgumentParser("01-random.py")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    if args.debug:
        enable_debug()

    main()
