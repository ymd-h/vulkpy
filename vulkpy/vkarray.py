import os
import functools
from typing import Iterable, Self

import numpy as np

from . import _vkarray

shader_dir = os.path.join(os.path.dirname(__file__), "shader")

class GPU:
    def __init__(self, idx: int=0, priority: float=0.0):
        """
        GPU

        Parameters
        ----------
        idx : int, optional
            Index to specify one GPU from multiple GPUs. Default is ``0``.
        priority : float, optional
            GPU priority. Default is ``0.0``.
        """
        self.gpu = _vkarray.GPU(idx, priority)

    @functools.cache
    def _createOp(self, spv: str,
                  local_size_x: int,
                  local_size_y: int,
                  local_size_z: int) -> _vkarray.Op:
        """
        Create 3-buffer Vector Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        local_size_x, local_size_y, local_size_z : int
            Subgroup size of compute shader

        Returns
        -------
        _vkarray.Op
           Operation
        """
        return self.gpu.createOp(spv, local_size_x, local_size_y, local_size_z)

    def _submit(self,
                spv: str,
                local_size_x: int, local_size_y: int, local_size_z: int,
                buffers: Iterable[_vkarray.Buffer],
                semaphores: Iterable[_vkarray.Semaphore]) -> _vkarray.Job:
        """
        Submit GPU Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        buffers : iterable of _vkarray.Buffer
            Buffers to be submitted.
        jobs : iterable of _vkarray.Job
            Depending Jobs to be waited.

        Returns
        -------
        _vkarray.Job
            Job
        """
        op = self._createOp(spv, local_size_x, local_size_y, local_size_z)
        size = buffers[0].size()
        return self.gpu.submit(op,
                               [b.info() for b in buffers],
                               _vkarray.DataShape(size, 1, 1),
                               _vkarray.VectorParams(size),
                               semaphores)


class Buffer:
    def __init__(self, gpu: GPU, *, data = None, shape = None):
        """
        Buffer for float (32bit)

        Parameters
        ----------
        gpu : GPU
            GPU instance to allocate at.
        data : array_like, optional
            Data which copy to GPU buffer.
        shape : array_like, optional
            Array shape

        Raises
        ------
        ValueError
            If both ``data`` and ``shape`` are ``None``.
        """
        self._gpu = gpu
        self._add = os.path.join(shader_dir, "add.spv")
        self._sub = os.path.join(shader_dir, "sub.spv")
        self._mul = os.path.join(shader_dir, "mul.spv")
        self._div = os.path.join(shader_dir, "div.spv")

        if data is not None:
            self.shape = np.asarray(data).shape
            self.buffer = self._gpu.gpu.toBuffer(data)
        elif shape is not None:
            self.shape = np.asarray(shape)
            self.buffer = self._gpu.gpu.createBuffer(int(self.shape.prod()))
        else:
            raise ValueError(f"`data` or `shape` must not be `None`.")

        self.array = np.asarray(self.buffer)
        self.array.shape = self.shape
        self.job = None

    def _op3(self, spv, other):
        if not np.array_equal(self.shape, other.shape):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        ret = Buffer(self._gpu, shape=self.shape)
        ret.job = self._gpu._submit(spv, 64, 1, 1,
                                    [self.buffer, other.buffer, ret.buffer],
                                    [b.job.getSemaphore() for b in [self, other]
                                     if b.job is not None])

        return ret

    def __add__(self, other: Self):
        return self._op3(self._add, other)

    def __sub__(self, other: Self):
        return self._op3(self._sub, other)

    def __mul__(self, other: Self):
        return self._op3(self._mul, other)

    def __truediv__(self, other: Self):
        return self._op3(self._div, other)

    def wait(self):
        """
        Wait Last Job
        """
        if self.job is not None:
            self.job.wait()

    def __getitem__(self, key):
        return self.array[key]
