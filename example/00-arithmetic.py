import numpy as np
import vulkpy as vk
from vulkpy.util import enable_debug

def main():
    enable_debug()

    gpu = vk.GPU()

    shape = (100,)
    a = vk.FloatBuffer(gpu, data=np.full(shape, 3))
    b = vk.FloatBuffer(gpr, data=np.full(shape, 5))

    c = a + b
    c.wait()
    print(c.array)

    d = c - a
    e = d - b
    e.wait()
    print(e.array)


if __name__ == "__main__":
    main()
