"""
vulkpy: GPGPU array on Vulkan
=============================

vulkpy provides GPU computations.

See Also
--------
vulkpy.vkarray : Core Module
vulkpy.random : Random Module


Examples
--------
>>> import vulkpy as vk

>>> gpu = vk.GPU()
>>> a = vk.Array(gpu, data=[1, 2, 3])
>>> b = vk.Array(gpu, data=[3, 3, 3])

>>> c = a + b
>>> print(c)
[4., 5., 6.]
"""
from .vkarray import GPU, Array
from . import random
