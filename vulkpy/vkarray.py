import os
import functools
from typing import Iterable, Self, Union

import numpy as np

from . import _vkarray

shader_dir = os.path.join(os.path.dirname(__file__), "shader")

Params = Union[_vkarray.VectorParams, _vkarray.VectorScalarParams]
Op = Union[_vkarray.OpVec2, _vkarray.OpVec3]

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
        self.gpu = _vkarray.createGPU(idx, priority)

    @functools.cache
    def _createOp(self, spv: str,
                  n: int,
                  params: Params,
                  local_size_x: int,
                  local_size_y: int,
                  local_size_z: int) -> Op:
        """
        Create GPU Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        n : int
            Number of buffers
        params : Params
            Parameters
        local_size_x, local_size_y, local_size_z : int
            Subgroup size of compute shader

        Returns
        -------
        std::shared_ptr<Op>
           Operation
        """
        return self.gpu.createOp(n, params,
                                 spv, local_size_x, local_size_y, local_size_z)

    def _submit(self,
                spv: str,
                local_size_x: int, local_size_y: int, local_size_z: int,
                buffers: Iterable[_vkarray.Buffer],
                shape: _vkarray.DataShape,
                params: Params,
                semaphores: Iterable[_vkarray.Semaphore]) -> _vkarray.Job:
        """
        Submit GPU Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        local_size_x, local_size_y, local_size_z : int
            Subgroup size of compute shader
        buffers : iterable of _vkarray.Buffer
            Buffers to be submitted.
        shape : _vkarray.DataShape
            Shape of data
        params : _vkarray.VectorParams, _vkarrayVectorScalarParams
            Parameters
        semaphores : iterable of _vkarray.Semaphore
            Depending Semaphores to be waited.

        Returns
        -------
        std::shared_ptr<_vkarray.Job>
            Job
        """
        op = self._createOp(spv, len(buffers), params,
                            local_size_x, local_size_y, local_size_z)
        size = buffers[0].size()
        return self.gpu.submit(op, [b.info() for b in buffers],
                               shape, params, semaphores)


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
        self._iadd = os.path.join(shader_dir, "iadd.spv")
        self._isub = os.path.join(shader_dir, "isub.spv")
        self._imul = os.path.join(shader_dir, "imul.spv")
        self._idiv = os.path.join(shader_dir, "idiv.spv")

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

    def _check_shape(self, other):
        if not np.array_equal(self.shape, other.shape):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

    def _opVec(self, spv, buffers):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 [b.buffer for b in buffers],
                                 _vkarray.DataShape(size, 1, 1),
                                 _vkarray.VectorParams(size),
                                 [b.job.getSemaphore() for b in buffers
                                  if b.job is not None])

    def _opVec3(self, spv, other):
        self._check_shape(other)
        ret = Buffer(self._gpu, shape=self.shape)
        ret.job = self._opVec(spv, [self, other, ret])
        return ret

    def _opVec2(self, spv, other):
        self._check_shape(other)
        self.job = self._opVec(spv, [self, other])

    def __add__(self, other: Self):
        return self._opVec3(self._add, other)

    def __sub__(self, other: Self):
        return self._opVec3(self._sub, other)

    def __mul__(self, other: Self):
        return self._opVec3(self._mul, other)

    def __truediv__(self, other: Self):
        return self._opVec3(self._div, other)

    def __iadd__(self, other: Self):
        self._opVec2(self._iadd, other)
        return self

    def __isub__(self, other: Self):
        self._opVec2(self._isub, other)
        return self

    def __imul__(self, other: Self):
        self._opVec2(self._isub, other)
        return self

    def __itruediv__(self, other: Self):
        self._opVec2(self._idiv, other)
        return self

    def wait(self):
        """
        Wait Last Job
        """
        if self.job is not None:
            self.job.wait()

    def __getitem__(self, key):
        return self.array[key]

    def __repr__(self):
        return f"<vulkpy.Buffer(shape={tuple(self.shape)})>"

    def __str__(self):
        return str(self.array)
