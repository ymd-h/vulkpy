"""
Random Module (:mod:`vulkpy.random`)
====================================

GPU-based Pseudo Random Number Generator (PRNG)


Examples
--------
>>> import vulkpy as vk
>>> gpu = vk.GPU()
>>> r = vk.random.Xoshiro128pp(gpu, seed=0)

[0, 1) uniform random numbers can be generated by
``random(shape=None, buffer=None)``.

>>> print(r.random(shape=(3,)))
[0.42977667 0.8235899  0.90622926]

Gaussian random numbers can be generated by
``normal(shape=None, buffer=None, mean=0.0, stddev=1.0)``.

>>> print(r.normal(shape=(3,)))
[-2.3403292  0.7247794  0.7118352]
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from . import _vkarray
from . import vkarray as vk
from .util import getShader

__all__ = ["Xoshiro128pp"]

class _ConvertMixin:
    _box_muller = getShader("prng_box_muller.spv")
    _ibox_muller = getShader("prng_ibox_muller.spv")

    def normal(self, *,
               shape: Optional[Iterable[int]] = None,
               buffer: Optional[vk.Array] = None,
               mean: float = 0.0,
               stddev: float = 1.0) -> vk.Array:
        """
        Generate Normal Distributing numbers

        Parameters
        ----------
        shape : iterable of ints, optional
            If specified, new ``vulkpy.Array`` with ``shape`` will be returned.
        buffer : vulkpy.Array
            If specified, generated numbers will be stored.

        Returns
        -------
        vulkpy.Array
            Array which will get random numbers.

        Raises
        ------
        ValueError
            If neither ``shape`` or ``buffer`` are specified

        Notes
        -----
        This method first generates [0, 1) uniform random numbers,
        then transforms them to normal distribution with Box-Muller method.
        Box-Muller might have problem in terms of random number quality,
        however, it is quite GPU friendly.
        """
        _local_size = 64
        if buffer is None:
            if shape is None:
                raise ValueError("One of `shape` and  `buffer` must be specified.")

            buffer = vk.Array(self._gpu, shape=shape)
        else:
            # For safety, we wait output buffer job.
            buffer.wait()

        n = int(np.prod(buffer.shape))
        floor_n = n // 2
        dshape = _vkarray.DataShape(floor_n, 1, 1)
        p = _vkarray.VectorScalar2Params(n, mean, stddev)
        if n % 2 == 0:
            # Even: Reuse `buffer`
            buffer = self.random(buffer=buffer)
            buffer.job = self._gpu._submit(self._ibox_muller,
                                           _local_size, 1, 1,
                                           [buffer], dshape, p)
        else:
            # Odd: Require additional space for intermediate [0, 1)
            rng = self.random(shape=2*(floor_n + 1))
            buffer.job = self._gpu._submit(self._box_muller,
                                           _local_size, 1, 1,
                                           [rng, buffer], dshape, p)

        return buffer

class Xoshiro128pp(_ConvertMixin):
    """
    xoshiro128++: Pseudo Random Number Generator

    Notes
    -----
    This class implements xoshiro128++ [1]_. Initial internal states are
    sequentially generated during construction on CPU and are spaced 2^64 steps.
    Generating (pseudo-)random numbers are executed parallelly on GPU.

    References
    ----------
    .. [1] S. Vigna "xoshiro / xoroshiro generators and the PRNG shootout",
       https://prng.di.unimi.it/
    """
    _spv = getShader("prng_xoshiro128pp.spv")

    def __init__(self, gpu: vk.GPU, size: int = 64, *, seed: Optional[int] = None):
        """
        Initialize Xoshiro128pp

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU where PRNG allocates
        size : int
            Number of internal states. These states generate random number parallelly.
        seed : int, optional
            Random seed. If ``None`` (default), use hardware random instead.
        """
        self._gpu = gpu

        if seed is None:
            self.rng = _vkarray.Xoshiro128pp(self._gpu.gpu, self._spv, size)
        else:
            self.rng = _vkarray.Xoshiro128pp(self._gpu.gpu, self._spv, size, seed)

    def random(self, *,
               shape: Optional[Iterable[int]] = None,
               buffer: Optional[vk.Array] = None) -> vk.Array:
        """
        Generate [0, 1) floating numbers

        Parameters
        ----------
        shape : iterable of ints, optional
            If specified, new ``vulkpy.Array`` with ``shape`` will be returned.
        buffer : vulkpy.Array
            If specified, generated numbers will be stored.

        Returns
        -------
        vulkpy.Array
            Array which will get random numbers.

        Raises
        ------
        ValueError
            If neither ``shape`` or ``buffer`` are specified.
        """
        if buffer is None:
            if shape is None:
                raise ValueError("One of `shape` and  `buffer` must be specified.")

            buffer = vk.Array(self._gpu, shape=shape)
        else:
            # For safety, we wait output buffer job.
            buffer.wait()

        n = int(np.prod(buffer.shape))
        buffer.job = self.rng.random(n, buffer.buffer.info())
        return buffer
