from __future__ import annotations

import os
import functools
from typing import Iterable, Optional, Self, Union

import numpy as np
import wblog

from .util import getShader
from . import _vkarray

__all__ = ["GPU", "Array"]

Params = Union[_vkarray.VectorParams,
               _vkarray.MultiVector2Params,
               _vkarray.VectorScalarParams,
               _vkarray.VectorScalar2Params,
               _vkarray.MatMulParams,
               _vkarray.AxisReductionParams]
Op = Union[_vkarray.OpVec1, _vkarray.OpVec2, _vkarray.OpVec3, _vkarray.OpVec4,
           _vkarray.Op2MultiVec2,
           _vkarray.OpVecScalar1, _vkarray.OpVecScalar2, _vkarray.OpVecScalar3,
           _vkarray.OpVec2Scalar1, _vkarray.OpVec2Scalar2,
           _vkarray.OpMatMul,
           _vkarray.OpAxisReduction]

logger = wblog.getLogger()

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
        self.canSubgroupArithmetic = self.gpu.canSubgroupArithmetic()

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
                jobs: Iterable[_vkarray.Job]) -> _vkarray.Job:
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
        jobs : iterable of _vkarray.Job
            Depending Semaphores to be waited.

        Returns
        -------
        _vkarray.Job
            Job
        """
        op = self._createOp(spv, len(buffers), params,
                            local_size_x, local_size_y, local_size_z)
        size = buffers[0].size()
        return self.gpu.submit(op, [b.info() for b in buffers], shape, params, jobs)

    def flush(self, arrays: Iterable[Array]):
        """
        Flush buffers

        Parameters
        ----------
        arrays : iterable of Array
            Arrays to be flushed
        """
        self.gpu.flush([a.buffer.range() for a in arrays])

    def wait(self):
        """
        Wait All GPU Operations
        """
        self.gpu.wait()


class Array:
    _add = getShader("add.spv")
    _sub = getShader("sub.spv")
    _mul = getShader("mul.spv")
    _div = getShader("div.spv")
    _iadd = getShader("iadd.spv")
    _isub = getShader("isub.spv")
    _imul = getShader("imul.spv")
    _idiv = getShader("idiv.spv")
    _add_scalar = getShader("add_scalar.spv")
    _sub_scalar = getShader("sub_scalar.spv")
    _mul_scalar = getShader("mul_scalar.spv")
    _div_scalar = getShader("div_scalar.spv")
    _iadd_scalar = getShader("iadd_scalar.spv")
    _isub_scalar = getShader("isub_scalar.spv")
    _imul_scalar = getShader("imul_scalar.spv")
    _idiv_scalar = getShader("idiv_scalar.spv")
    _rsub_scalar = getShader("rsub_scalar.spv")
    _rdiv_scalar = getShader("rdiv_scalar.spv")
    _matmul = getShader("matmul.spv")
    _max = getShader("max.spv")
    _min = getShader("min.spv")
    _imax = getShader("imax.spv")
    _imin = getShader("imin.spv")
    _max_scalar = getShader("max_scalar.spv")
    _min_scalar = getShader("min_scalar.spv")
    _imax_scalar = getShader("imax_scalar.spv")
    _imin_scalar = getShader("imin_scalar.spv")
    _abs = getShader("abs.spv")
    _sign = getShader("sign.spv")
    _iabs = getShader("iabs.spv")
    _isign = getShader("isign.spv")
    _sin = getShader("sin.spv")
    _cos = getShader("cos.spv")
    _tan = getShader("tan.spv")
    _isin = getShader("isin.spv")
    _icos = getShader("icos.spv")
    _itan = getShader("itan.spv")
    _asin = getShader("asin.spv")
    _acos = getShader("acos.spv")
    _atan = getShader("atan.spv")
    _iasin = getShader("iasin.spv")
    _iacos = getShader("iacos.spv")
    _iatan = getShader("iatan.spv")
    _sinh = getShader("sinh.spv")
    _cosh = getShader("cosh.spv")
    _tanh = getShader("tanh.spv")
    _isinh = getShader("isinh.spv")
    _icosh = getShader("icosh.spv")
    _itanh = getShader("itanh.spv")
    _asinh = getShader("asinh.spv")
    _acosh = getShader("acosh.spv")
    _atanh = getShader("atanh.spv")
    _iasinh = getShader("iasinh.spv")
    _iacosh = getShader("iacosh.spv")
    _iatanh = getShader("iatanh.spv")
    _exp = getShader("exp.spv")
    _log = getShader("log.spv")
    _iexp = getShader("iexp.spv")
    _ilog = getShader("ilog.spv")
    _exp2 = getShader("exp2.spv")
    _log2 = getShader("log2.spv")
    _iexp2 = getShader("iexp2.spv")
    _ilog2 = getShader("ilog2.spv")
    _sqrt = getShader("sqrt.spv")
    _invsqrt = getShader("invsqrt.spv")
    _isqrt = getShader("isqrt.spv")
    _iinvsqrt = getShader("iinvsqrt.spv")
    _pow = getShader("pow.spv")
    _ipow = getShader("ipow.spv")
    _pow_scalar = getShader("pow_scalar.spv")
    _ipow_scalar = getShader("ipow_scalar.spv")
    _rpow_scalar = getShader("rpow_scalar.spv")
    _clamp = getShader("clamp.spv")
    _iclamp = getShader("iclamp.spv")
    _clamp_sv = getShader("clamp_sv.spv")
    _iclamp_sv = getShader("iclamp_sv.spv")
    _clamp_vs = getShader("clamp_vs.spv")
    _iclamp_vs = getShader("iclamp_vs.spv")
    _clamp_ss = getShader("clamp_ss.spv")
    _iclamp_ss = getShader("iclamp_ss.spv")
    _sum = getShader("sum.spv")
    _sum_v1_3 = getShader("sum_v1.3.spv")
    _sum_axis = getShader("sum_axis.spv")

    def __init__(self, gpu: GPU, *, data = None, shape = None):
        """
        Array for float (32bit)

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

        if data is not None:
            self.shape = np.asarray(data).shape
            self.buffer = self._gpu.gpu.toBuffer(np.ravel(data))
        elif shape is not None:
            self.shape = np.asarray(shape, dtype=int)
            self.buffer = self._gpu.gpu.createBuffer(int(self.shape.prod()))
        else:
            raise ValueError(f"`data` or `shape` must not be `None`.")

        self.array = np.asarray(self.buffer)
        self.array.shape = self.shape
        self.job = None

    def __del__(self):
        self.wait()

    def _check_shape(self, other):
        if not np.array_equal(self.shape, other.shape):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

    def _opVec(self, spv, buffers):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 [b.buffer for b in buffers],
                                 _vkarray.DataShape(size, 1, 1),
                                 _vkarray.VectorParams(size),
                                 [b.job for b in buffers if b.job is not None])

    def _opVec3(self, spv, other):
        self._check_shape(other)
        ret = Array(self._gpu, shape=self.shape)
        ret.job = self._opVec(spv, [self, other, ret])
        return ret

    def _opVec2(self, spv, other=None):
        if other is not None:
            self._check_shape(other)
            self.job = self._opVec(spv, [self, other])
        else:
            ret = Array(self._gpu, shape=self.shape)
            ret.job = self._opVec(spv, [self, ret])
            return ret

    def _opVec1(self, spv):
        self.job = self._opVec(spv, [self])

    def _opVecScalar(self, spv, buffers, scalar):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 [b.buffer for b in buffers],
                                 _vkarray.DataShape(size, 1, 1),
                                 _vkarray.VectorScalarParams(size, scalar),
                                 [b.job for b in buffers if b.job is not None])

    def _opVecScalar2(self, spv, other):
        ret = Array(self._gpu, shape=self.shape)
        ret.job = self._opVecScalar(spv, [self, ret], other)
        return ret

    def _opVecScalar1(self, spv, other):
        self.job = self._opVecScalar(spv, [self], other)

    def _opVec2Scalar(self, spv, buffers, scalars):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 [b.buffer for b in buffers],
                                 _vkarray.DataShape(size, 1, 1),
                                 _vkarray.VectorScalar2Params(size, *scalars),
                                 [b.job for b in buffers if b.job is not None])

    def __add__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            return self._opVec3(self._add, other)
        else:
            return self._opVecScalar2(self._add_scalar, other)

    def __sub__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            return self._opVec3(self._sub, other)
        else:
            return self._opVecScalar2(self._sub_scalar, other)

    def __mul__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            return self._opVec3(self._mul, other)
        else:
            return self._opVecScalar2(self._mul_scalar, other)

    def __truediv__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            return self._opVec3(self._div, other)
        else:
            return self._opVecScalar2(self._div_scalar, other)

    def __iadd__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            self._opVec2(self._iadd, other)
        else:
            self._opVecScalar1(self._iadd_scalar, other)
        return self

    def __isub__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            self._opVec2(self._isub, other)
        else:
            self._opVecScalar1(self._isub_scalar, other)
        return self

    def __imul__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            self._opVec2(self._imul, other)
        else:
            self._opVecScalar1(self._imul_scalar, other)
        return self

    def __itruediv__(self, other: Union[Self, float]) -> Array:
        if isinstance(other, Array):
            self._opVec2(self._idiv, other)
        else:
            self._opVecScalar1(self._idiv_scalar, other)
        return self

    def __radd__(self, other: float) -> Array:
        return self._opVecScalar2(self._add_scalar, other)

    def __rsub__(self, other: float) -> Array:
        return self._opVecScalar2(self._rsub_scalar, other)

    def __rmul__(self, other: float) -> Array:
        return self._opVecScalar2(self._mul_scalar, other)

    def __rtruediv__(self, other: float) -> Array:
        return self._opVecScalar2(self._rdiv_scalar, other)

    def __matmul__(self, other: Array) -> Array:
        if ((len(self.shape) > 3) or
            (len(other.shape) > 3) or
            (self.shape[-1] != other.shape[0])):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        shape = tuple(self.shape)[:-1] + tuple(other.shape)[1:]
        if len(shape) == 0:
            shape = (1,)

        rowA = self.shape[0] if len(self.shape) > 1 else 1
        contractSize = self.shape[-1]
        columnB = other.shape[1] if len(other.shape) > 1 else 1

        ret = Array(self._gpu, shape=shape)
        ret.job = self._gpu._submit(self._matmul, 1, 64, 1,
                                    [self.buffer, other.buffer, ret.buffer],
                                    _vkarray.DataShape(rowA, columnB, 1),
                                    _vkarray.MatMulParams(rowA,contractSize,columnB),
                                    [b.job for b in [self, other]
                                     if b.job is not None])
        return ret

    def wait(self):
        """
        Wait Last Job
        """
        if self.job is not None:
            self.job.wait()
            self.job = None

    def flush(self):
        """
        Flush Buffer to GPU
        """
        self._gpu.flush([self])

    def __getitem__(self, key) -> Union[float, np.ndarray]:
        self.wait()
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self) -> str:
        return f"<vulkpy.Buffer(shape={tuple(self.shape)})>"

    def __str__(self) -> str:
        self.wait()
        return str(self.array)

    def __array__(self) -> np.ndarray:
        self.wait()
        return self.array

    def reshape(self, shape: tuple[int]):
        """
        Reshape of this array

        Parameters
        ----------
        shape : tuple of int
            New shape

        Raises
        ------
        ValueError
            If ``shape`` is incompatible
        """
        self.array.shape = shape
        self.shape = self.array.shape

    def max(self, other: Union[Array, float],
            inplace: bool = False) -> Optional[Array]:
        """
        Element-wise Max

        Parameters
        ----------
        other : Array or float
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.

        Raises
        ------
        ValueError
            If shape is not same.
        """
        if isinstance(other, Array):
            if inplace:
                self._opVec2(self._imax, other)
            else:
                return self._opVec3(self._max, other)
        else:
            if inplace:
                self._opVecScalar1(self._imax_scalar, other)
            else:
                return self._opVecScalar2(self._max_scalar, other)

    def min(self, other: Union[Array, float],
            inplace: bool = False) -> Optional[Array]:
        """
        Element-wise Min

        Parameters
        ----------
        other : Array or float
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.

        Raises
        ------
        ValueError
            If shape is not same.
        """
        if isinstance(other, Array):
            if inplace:
                self._opVec2(self._imin, other)
            else:
                return self._opVec3(self._min, other)
        else:
            if inplace:
                self._opVecScalar1(self._imin_scalar, other)
            else:
                return self._opVecScalar2(self._min_scalar, other)

    def abs(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise Abs

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iabs)
        else:
            return self._opVec2(self._abs)

    def sign(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise sign

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._isign)
        else:
            return self._opVec2(self._sign)

    def sin(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise sin()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._isin)
        else:
            return self._opVec2(self._sin)

    def cos(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise cos()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._icos)
        else:
            return self._opVec2(self._cos)

    def tan(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise tan()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._itan)
        else:
            return self._opVec2(self._tan)

    def asin(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise asin()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iasin)
        else:
            return self._opVec2(self._asin)

    def acos(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise acos()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iacos)
        else:
            return self._opVec2(self._acos)

    def atan(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise atan()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iatan)
        else:
            return self._opVec2(self._atan)

    def sinh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise sinh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._isinh)
        else:
            return self._opVec2(self._sinh)

    def cosh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise cosh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._icosh)
        else:
            return self._opVec2(self._cosh)

    def tanh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise tanh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._itanh)
        else:
            return self._opVec2(self._tanh)

    def asinh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise asinh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iasinh)
        else:
            return self._opVec2(self._asinh)

    def acosh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise acosh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iacosh)
        else:
            return self._opVec2(self._acosh)

    def atanh(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise atanh()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iatanh)
        else:
            return self._opVec2(self._atanh)

    def exp(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise exp()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iexp)
        else:
            return self._opVec2(self._exp)

    def log(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise log()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._ilog)
        else:
            return self._opVec2(self._log)

    def exp2(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise exp2()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iexp2)
        else:
            return self._opVec2(self._exp2)

    def log2(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise log2()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._ilog2)
        else:
            return self._opVec2(self._log2)

    def sqrt(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise sqrt()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._isqrt)
        else:
            return self._opVec2(self._sqrt)

    def invsqrt(self, inplace: bool = False) -> Optional[Array]:
        """
        Element-wise 1/sqrt()

        Parameters
        ----------
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        if inplace:
            self._opVec1(self._iinvsqrt)
        else:
            return self._opVec2(self._invsqrt)

    def __pow__(self, other: Union[Array, float]) -> Array:
        if isinstance(other, Array):
            return self._opVec3(self._pow, other)
        else:
            return self._opVecScalar2(self._pow_scalar, other)

    def __ipow__(self, other: Union[Array, float]) -> Array:
        if isinstance(other, Array):
            self._opVec2(self._ipow, other)
        else:
            self._opVecScalar1(self._ipow_scalar, other)
        return self

    def __rpow__(self, other: float) -> Array:
        return self._opVecScalar2(self._rpow_scalar, other)

    def clamp(self, min: Union[Array, float], max: Union[Array, float],
              inplace: bool = False) -> Optional[Array]:
        """
        Element-wise clamp()

        Parameters
        ----------
        min, max : Array or float
            Minimum/Maximum value
        inplace : bool
            If ``True``, update inplace, otherwise returns new array.
            Default value is ``False``.

        Returns
        -------
        None
            When ``replace=True``.
        Array
            When ``replace=False``.
        """
        min_is_array = isinstance(min, Array)
        if min_is_array:
            self._check_shape(min)

        max_is_array = isinstance(max, Array)
        if max_is_array:
            self._check_shape(max)

        if not inplace:
            ret = Array(self._gpu, shape=self.shape)
            if min_is_array and max_is_array:
                ret.job = self._opVec(self._clamp, [self, min, max, ret])
            elif max_is_array:
                ret.job = self._opVecScalar(self._clamp_sv, [self, max, ret], min)
            elif min_is_array:
                ret.job = self._opVecScalar(self._clamp_vs, [self, min, ret], max)
            else:
                ret.job = self._opVec2Scalar(self._clamp_ss, [self, ret], [min, max])
            return ret
        else:
            # inplace
            if min_is_array and max_is_array:
                self.job = self._opVec(self._iclamp, [self, min, max])
            elif max_is_array:
                self.job = self._opVecScalar(self._iclamp_sv, [self, max], min)
            elif min_is_array:
                self.job = self._opVecScalar(self._iclamp_vs, [self, min], max)
            else:
                self.job = self._opVec2Scalar(self._iclamp_ss, [self], [min, max])


    def _axis_reduction(self, spv, axis):
        # Ensure axis is flattened decending unique indices set.
        axis = np.unique(axis, axis=None)[::-1]

        tmp = self
        # Loop from last axis, to keep previous axis indices.
        for a in axis:
            prev_prod = int(np.prod(tmp.shape[:a]))
            axis_size = int(tmp.shape[a])
            post_prod = int(np.prod(tmp.shape[a+1:]))

            ret = Array(self._gpu, shape=np.concatenate((tmp.shape[:a],
                                                         tmp.shape[a+1:]),
                                                        axis=0))
            jobs = [] if tmp.job is None else [tmp.job]
            ret.job = self._gpu._submit(spv, 1, 64, 1,
                                        [tmp.buffer, ret.buffer],
                                        _vkarray.DataShape(prev_prod, post_prod, 1),
                                        _vkarray.AxisReductionParams(prev_prod,
                                                                     axis_size,
                                                                     post_prod),
                                        jobs)
            tmp = ret

        return ret

    def sum(self, axis: Union[int, Iterable[int]]=None):
        """
        Summarize

        Parameters
        ----------
        axis : int, optional
            Reduction array

        Returns
        -------
        vulkpy.Array
            Summarized array
        """
        if axis is None:
            _local_size = 64
            if self._gpu.canSubgroupArithmetic:
                logger.info("sum w/ SubgroupArithmetic")
                f = lambda tmp, ret: self._opVec(self._sum_v1_3, [tmp, ret])
            else:
                logger.info("sum w/o SubgroupArithmetic")
                def f(tmp, ret):
                    b = [tmp.buffer, ret.buffer]
                    p = _vkarray.MultiVector2Params(*[bb.size() for bb in b])
                    return self._gpu._submit(self._sum, _local_size, 1, 1,
                                             b, _vkarray.DataShape(64,1,1), p,
                                             [tmp.job] if tmp.job else [])

            n = self.buffer.size()
            tmp = self

            while True:
                m = (n // _local_size) + ((n % _local_size) != 0)
                ret = Array(self._gpu, shape=(m,))
                ret.job = f(tmp, ret)

                if m == 1:
                    return ret

                n = m
                tmp = ret
        else:
            return self._axis_reduction(self._sum_axis, axis)
