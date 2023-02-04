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

    def _submitVec3(self, spv: str, buffers: Iterable[FloatBuffer]) -> _vkarray.Job:
        """
        Submit 3-buffer Vector Operation

        Parameters
        ----------
        spv : str
            Compute Shader file name of SPIR-V (.spv)
        buffers : iterable of FloatBuffer
            Buffers to be submitted.

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
                                   _vkarray.VectorParams(size))


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
        self.gpu = gpu
        self.spv = os.path.join(shader_dir, "add.spv")

        if data is not None:
            self.shape = np.asarray(data).shape
            self.buffer = self.gpu.gpu.toFloatBuffer(data)
        elif shape is not None:
            self.shape = np.asarray(shape)
            self.buffer = self.gpu.gpu.createFloatBuffer(int(self.shape.prod()))
        else:
            raise ValueError(f"`data` or `shape` must not be `None`.")


    def __add__(self, other: FloatBuffer):
        if self.shape != other.shape:
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        ret = FloatBuffer(self.gpu, shape=self.shape)
        job = self.gpu._submitVec3(self.spv, [self.buffer, other.buffer, ret.buffer])

        return ret, job
