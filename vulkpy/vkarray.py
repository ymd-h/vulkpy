import os
import functools
import numpy as np

import _vkarray

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
    def _createOpVec3(self, spv: str) -> _vkarray.Vec3Op:
        """
        Create 3-buffer Vector Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)

        Returns
        -------
        _vkarray.Vec3Op
           Operation
        """
        return self.gpu.createOpVec3(spv, 64, 1, 1)

    def _submitVec3(self,
                    spv: str,
                    buffers: Iterable[_vkarray.FloatBuffer]) -> _vkarray.Job:
        """
        Submit 3-buffer Vector Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        buffers : iterable of _vkarray.FloatBuffer
            Buffers to be submitted.
        jobs : iterable of _vkarray.Job
            Depending Jobs to be waited.

        Returns
        -------
        _vkarray.Job
            Job
        """
        op = self._createOpVec3(spv)
        size = int(buffers[0].prod())
        return self.gpu.submitVec3(op,
                                   [b.info() for b in buffers],
                                   _vkarray.DataShape(size, 1, 1),
                                   _vkarray.VectorParams(size),
                                   [job.getSemaphore() for job in jobs])


class FloatBuffer:
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
            self.buffer = self._gpu.gpu.toFloatBuffer(data)
        elif shape is not None:
            self.shape = np.asarray(shape)
            self.buffer = self._gpu.gpu.createFloatBuffer(int(self.shape.prod()))
        else:
            raise ValueError(f"`data` or `shape` must not be `None`.")

        self.array = np.view(np.asarray(self.buffer), self.shape)
        self.job = None

    def _op3(self, spv, other):
        if np.array_equal(self.shape, other.shape):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        ret = FloatBuffer(self._gpu, shape=self.shape)
        self.job = self._gpu._submitVec3(spv,
                                         [self.buffer, other.buffer, ret.buffer],
                                         [b.job for b in [self, other]
                                          if b.job is not None])

        return ret

    def __add__(self, other: FloatBuffer):
        return self._op3(self._add, other)

    def __sub__(self, other: FloatBuffer):
        return self._op3(self._sub, other)

    def __mul__(self, other: FloatBuffer):
        return self._op3(self._mul, other)

    def __truediv__(self, other: FloatBuffer):
        return self._op3(self._div, other)
