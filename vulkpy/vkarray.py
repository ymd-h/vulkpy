"""
vulkpy Core Module (:mod:`vulkpy.vkarray`)
==========================================


* GPU
* Array
"""

from __future__ import annotations

import os
import functools
from typing import Iterable, List, Optional, Union

import numpy as np
import wblog

from .util import getShader
from ._vkarray import createGPU, DataShape, Job
from ._vkarray import (
    VectorParams,
    MultiVector2Params,
    VectorScalarParams,
    VectorScalar2Params,
    MatMulParams,
    AxisReductionParams,
    BroadcastParams,
    Multi3BroadcastParams,
    BatchAffineParams,
    AxisGatherParams,
)

__all__ = [
    "GPU",
    "U32Array",
    "Array"
]

Params = Union[
    VectorParams,
    MultiVector2Params,
    VectorScalarParams,
    VectorScalar2Params,
    MatMulParams,
    AxisReductionParams,
    BroadcastParams,
    Multi3BroadcastParams,
    BatchAffineParams,
]

logger = wblog.getLogger()

class GPU:
    """
    GPU instance
    """
    def __init__(self, idx: int=0, priority: float=0.0):
        """
        Initialize GPU

        Parameters
        ----------
        idx : int, optional
            Index to specify one GPU from multiple GPUs. Default is ``0``.
        priority : float, optional
            GPU priority. Default is ``0.0``.
        """
        self.gpu = createGPU(idx, priority)
        self.canSubgroupArithmetic = self.gpu.canSubgroupArithmetic()
        logger.info(f"GPU {idx}: Subgroup Arithmetic: {self.canSubgroupArithmetic}")

    def _submit(self,
                spv: str,
                local_size_x: int, local_size_y: int, local_size_z: int,
                arrays: Iterable[Array],
                shape: DataShape,
                params: Params) -> Job:
        infos = [a.buffer.info() for a in arrays]
        jobs = [a.job for a in arrays if a.job is not None]
        return self.gpu.submit(spv, local_size_x, local_size_y, local_size_z,
                               infos, shape, params, jobs)

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


KeyType = Union[int, np.ndarray]
ValueType = Union[int, float, np.ndarray]


class _GPUArray:
    def __init__(self, gpu: GPU):
        self._gpu: GPU = gpu

        # Pipeline job to write this Array.
        self.job: Optional[Job] = None

        # Hold temporary resources until pipeline job finish
        # to avoid freeing memories in use.
        self._keep: List[Union[Shape, Array]] = []

    def __del__(self):
        self.wait()

    def wait(self):
        """
        Wait Last Job
        """
        if self.job is not None:
            self.job.wait()
            self.job = None

        self._keep = []

    def flush(self):
        """
        Flush Buffer to GPU
        """
        self._gpu.flush([self])

    def __getitem__(self, key: KeyType) -> ValueType:
        self.wait()
        return self.array[key]

    def __setitem__(self, key: KeyType, value: ValueType):
        self.array[key] = value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(shape={tuple(self.shape)})>"

    def __str__(self) -> str:
        self.wait()
        return str(self.array)

    def __array__(self) -> np.ndarray:
        self.wait()
        return self.array


class U32Array(_GPUArray):
    """
    GPU Array of uint (32bit) for shape or indices
    """
    def __init__(self, gpu: GPU, *,
                 data: Optional[Iterable[int]] = None,
                 shape: Optional[Iterable[int]] = None):
        super().__init__(gpu)

        if data is not None:
            data = np.asarray(data, dtype=np.uint32)
            self.shape = data.shape
            self.buffer = self._gpu.gpu.toU32Buffer(np.ravel(data))
        else:
            if shape is None:
                raise ValueError("One of `data` or `shape` must be specified.")

            self.shape = np.asarray(shape, dtype=int)
            self.buffer = self._gpu.gpu.createU32Buffer(np.prod(self.shape))

        self.array = np.asarray(self.buffer)
        self.array.shape = self.shape

    def to_onehot(self, num_classes: int) -> Array:
        """
        Convert to one hot vector

        Parameters
        ----------
        num_classes : int
            Number of classes

        Returns
        -------
        vulkpy.Array
            One hot vector
        """
        return Array(self._gpu, data=np.identity(num_classes)).gather(self, axis=0)


class Shape(U32Array):
    """
    GPU Array of uint (32bit) for shape
    """
    def __init__(self, gpu: GPU, *,
                 data: Optional[Iterable[int]] = None,
                 ndim: Optional[int] = None):
        """
        Initialize Shape

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        data : iterable of ints, optional
            Data
        ndim : int, optional
            ndim
        """
        super().__init__(gpu, data=data, shape=(ndim,) if ndim is not None else None)


class Array(_GPUArray):
    """
    GPU Array for float (32bit)
    """
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
    _add_broadcast = getShader("add_broadcast.spv")
    _sub_broadcast = getShader("sub_broadcast.spv")
    _mul_broadcast = getShader("mul_broadcast.spv")
    _div_broadcast = getShader("div_broadcast.spv")
    _iadd_broadcast = getShader("iadd_broadcast.spv")
    _isub_broadcast = getShader("isub_broadcast.spv")
    _imul_broadcast = getShader("imul_broadcast.spv")
    _idiv_broadcast = getShader("idiv_broadcast.spv")
    _matmul = getShader("matmul.spv")
    _max = getShader("max.spv")
    _min = getShader("min.spv")
    _imax = getShader("imax.spv")
    _imin = getShader("imin.spv")
    _max_scalar = getShader("max_scalar.spv")
    _min_scalar = getShader("min_scalar.spv")
    _imax_scalar = getShader("imax_scalar.spv")
    _imin_scalar = getShader("imin_scalar.spv")
    _max_broadcast = getShader("max_broadcast.spv")
    _min_broadcast = getShader("min_broadcast.spv")
    _imax_broadcast = getShader("imax_broadcast.spv")
    _imin_broadcast = getShader("imin_broadcast.spv")
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
    _pow_broadcast = getShader("pow_broadcast.spv")
    _ipow_broadcast = getShader("ipow_broadcast.spv")
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
    _sum_axis_rebroadcast = getShader("sum_axis_rebroadcast.spv")
    _prod = getShader("prod.spv")
    _prod_v1_3 = getShader("prod_v1.3.spv")
    _prod_axis = getShader("prod_axis.spv")
    _prod_axis_rebroadcast = getShader("prod_axis_rebroadcast.spv")
    _maximum = getShader("maximum.spv")
    _maximum_v1_3 = getShader("maximum_v1.3.spv")
    _maximum_axis = getShader("maximum_axis.spv")
    _maximum_axis_rebroadcast = getShader("maximum_axis_rebroadcast.spv")
    _minimum = getShader("minimum.spv")
    _minimum_v1_3 = getShader("minimum_v1.3.spv")
    _minimum_axis = getShader("minimum_axis.spv")
    _minimum_axis_rebroadcast = getShader("minimum_axis_rebroadcast.spv")
    _broadcast = getShader("broadcast.spv")
    _gather = getShader("gather.spv")
    _gather_axis = getShader("gather_axis.spv")

    def __init__(self, gpu: GPU, *, data = None, shape = None):
        """
        Initialize Array

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
        super().__init__(gpu)

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

    def _check_shape(self, other):
        if not np.array_equal(self.shape, other.shape):
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

    def _opVec(self, spv, arrays):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 arrays,
                                 DataShape(size, 1, 1),
                                 VectorParams(size))

    def _opVec3(self, spv, other):
        self._check_shape(other)
        ret = Array(self._gpu, shape=self.shape)
        ret.job = self._opVec(spv, [self, other, ret])
        ret._keep = [self, other]
        return ret

    def _opVec2(self, spv, other=None):
        if other is not None:
            self._check_shape(other)
            self.job = self._opVec(spv, [self, other])
            self._keep = [other]
        else:
            ret = Array(self._gpu, shape=self.shape)
            ret.job = self._opVec(spv, [self, ret])
            ret._keep = [self]
            return ret

    def _opVec1(self, spv):
        self.job = self._opVec(spv, [self])

    def _opVecScalar(self, spv, arrays, scalar):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 arrays,
                                 DataShape(size, 1, 1),
                                 VectorScalarParams(size, scalar))

    def _opVecScalar2(self, spv, other):
        ret = Array(self._gpu, shape=self.shape)
        ret.job = self._opVecScalar(spv, [self, ret], other)
        ret._keep = [self]
        return ret

    def _opVecScalar1(self, spv, other):
        self.job = self._opVecScalar(spv, [self], other)

    def _opVec2Scalar(self, spv, arrays, scalars):
        size = self.buffer.size()
        return self._gpu._submit(spv, 64, 1, 1,
                                 arrays,
                                 DataShape(size, 1, 1),
                                 VectorScalar2Params(size, *scalars))

    def _op(self, other, spv, spv_scalar, spv_broadcast):
        if not isinstance(other, Array):
            return self._opVecScalar2(spv_scalar, other)
        if np.array_equal(self.shape, other.shape):
            ret = self._opVec3(spv, other)
            return ret

        shape = np.broadcast_shapes(self.shape, other.shape)
        ndim = len(shape)

        shapeABC = Shape(self._gpu, ndim=3*ndim)
        shapeABC[:] = 1
        shapeABC[  ndim- self.array.ndim:  ndim] = self.shape
        shapeABC[2*ndim-other.array.ndim:2*ndim] = other.shape
        shapeABC[2*ndim                 :      ] = shape
        shapeABC.flush()

        ret = Array(self._gpu, shape=shape)
        ret.job = self._gpu._submit(spv_broadcast, 64, 1, 1,
                                    [self, other, ret, shapeABC],
                                    DataShape(ret.buffer.size(), 1, 1),
                                    Multi3BroadcastParams(self.buffer.size(),
                                                          other.buffer.size(),
                                                          ret.buffer.size(),
                                                          ndim))

        ret._keep = [self, other, shapeABC]
        return ret

    def __add__(self, other: Union[Array, float]) -> Array:
        return self._op(other, self._add, self._add_scalar, self._add_broadcast)

    def __sub__(self, other: Union[Array, float]) -> Array:
        return self._op(other, self._sub, self._sub_scalar, self._sub_broadcast)

    def __mul__(self, other: Union[Array, float]) -> Array:
        return self._op(other, self._mul, self._mul_scalar, self._mul_broadcast)

    def __truediv__(self, other: Union[Array, float]) -> Array:
        return self._op(other, self._div, self._div_scalar, self._div_broadcast)

    def _iop(self, other, spv, spv_scalar, spv_broadcast):
        if not isinstance(other, Array):
            self._opVecScalar1(spv_scalar, other)
        elif np.array_equal(self.shape, other.shape):
            self._opVec2(spv, other)
        else:
            shape = np.broadcast_shapes(self.shape, other.shape)
            if not np.array_equal(shape, self.shape):
                raise ValueError(f"Incompatible shape. {shape} vs {self.shape}")
            ndim = shape[0]

            shapeAB = Shape(self._gpu, ndim=2*ndim)
            shapeAB[:] = 1
            shapeAB[:ndim] = shape
            if other.array.ndim > 0:
                shapeAB[-other.array.ndim:] = other.shape
            shapeAB.flush()

            self.job = self._gpu._submit(spv_broadcast, 64, 1, 1,
                                         [self, other, shapeAB],
                                         DataShape(self.buffer.size(), 1, 1),
                                         BroadcastParams(self.buffer.size(),
                                                         other.buffer.size(),
                                                         ndim))
            self._keep = [other, shapeAB]

        return self

    def __iadd__(self, other: Union[Array, float]) -> Array:
        return self._iop(other, self._iadd, self._iadd_scalar, self._iadd_broadcast)

    def __isub__(self, other: Union[Array, float]) -> Array:
        return self._iop(other, self._isub, self._isub_scalar, self._isub_broadcast)

    def __imul__(self, other: Union[Array, float]) -> Array:
        return self._iop(other, self._imul, self._imul_scalar, self._imul_broadcast)

    def __itruediv__(self, other: Union[Array, float]) -> Array:
        return self._iop(other, self._idiv, self._idiv_scalar, self._idiv_broadcast)

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
                                    [self, other, ret],
                                    DataShape(rowA, columnB, 1),
                                    MatMulParams(rowA,contractSize,columnB))
        ret._keep = [self, other]
        return ret

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
        if not inplace:
            return self._op(other, self._max, self._max_scalar, self._max_broadcast)
        self._iop(other, self._imax, self._imax_scalar, self._imax_broadcast)

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
        if not inplace:
            return self._op(other, self._min, self._min_scalar, self._min_broadcast)
        self._iop(other, self._imin, self._imin_scalar, self._imin_broadcast)

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
        return self._op(other, self._pow, self._pow_scalar, self._pow_broadcast)

    def __ipow__(self, other: Union[Array, float]) -> Array:
        self._iop(other, self._ipow, self._ipow_scalar, self._ipow_broadcast)
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
        shape_set = [self.shape]
        min_is_array = isinstance(min, Array)
        if min_is_array:
            shape_set.append(min.shape)

        max_is_array = isinstance(max, Array)
        if max_is_array:
            shape_set.append(max.shape)

        _s = self
        if len(shape_set) > 1:
            shape = np.broadcast_shapes(*shape_set)

            if not np.array_equal(shape, self.shape):
                if inplace:
                    raise ValueError(f"Incompatible shape")
                _s = self.broadcast_to(shape)

            if min_is_array and not np.array_equal(shape, min.shape):
                min = min.broadcast_to(shape)

            if max_is_array and not np.array_equal(shape, max.shape):
                max = max.broadcast_to(shape)

        if not inplace:
            ret = Array(self._gpu, shape=_s.shape)
            if min_is_array and max_is_array:
                ret.job = _s._opVec(self._clamp, [_s, min, max, ret])
                ret._keep = [_s, min, max]
            elif max_is_array:
                ret.job = _s._opVecScalar(self._clamp_sv, [_s, max, ret], min)
                ret._keep = [_s, max]
            elif min_is_array:
                ret.job = _s._opVecScalar(self._clamp_vs, [_s, min, ret], max)
                ret._keep = [_s, min]
            else:
                ret.job = _s._opVec2Scalar(self._clamp_ss, [_s, ret], [min, max])
                ret._keep = [_s]
            return ret
        else:
            # inplace
            if min_is_array and max_is_array:
                self.job = self._opVec(self._iclamp, [self, min, max])
                self._keep = [min, max]
            elif max_is_array:
                self.job = self._opVecScalar(self._iclamp_sv, [self, max], min)
                self._keep = [max]
            elif min_is_array:
                self.job = self._opVecScalar(self._iclamp_vs, [self, min], max)
                self._keep = [min]
            else:
                self.job = self._opVec2Scalar(self._iclamp_ss, [self], [min, max])
                self._keep = []


    def _axis_reduction(self, spv, axis, keepdims):
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
            ret.job = self._gpu._submit(spv, 1, 64, 1,
                                        [tmp, ret],
                                        DataShape(prev_prod, post_prod, 1),
                                        AxisReductionParams(prev_prod,
                                                            axis_size,
                                                            post_prod))
            ret._keep = [tmp]
            tmp = ret

        if keepdims:
            shape = np.array(self.shape)
            for a in axis:
                shape[a] = 1
            ret.reshape(shape)
        return ret

    def _reduce(self,
                spv, spv_v1_3, spv_axis, spv_rebroadcast,
                axis, keepdims, rebroadcast):
        if rebroadcast:
            if not isinstance(axis, int):
                raise ValueError("When `rebroadcast` is specified, " +
                                 "`axis` must be `int`")

            prev_prod = int(np.prod(self.shape[:axis]))
            axis_size = int(self.shape[axis])
            post_prod = int(np.prod(self.shape[axis+1:]))

            ret = Array(self._gpu, shape=self.shape)
            ret.job = self._gpu._submit(spv_rebroadcast, 1, 64, 1,
                                        [self, ret],
                                        DataShape(prev_prod, post_prod, 1),
                                        AxisReductionParams(prev_prod,
                                                            axis_size,
                                                            post_prod))
            ret._keep = [self]
            return ret

        if axis is None:
            _local_size = 64
            if self._gpu.canSubgroupArithmetic:
                f = lambda tmp, ret: self._opVec(spv_v1_3, [tmp, ret])
            else:
                def f(tmp, ret):
                    b = [tmp, ret]
                    p = MultiVector2Params(*[bb.buffer.size() for bb in b])
                    return self._gpu._submit(spv, _local_size, 1, 1,
                                             b, DataShape(64,1,1), p)

            n = self.buffer.size()
            tmp = self

            while True:
                m = (n // _local_size) + ((n % _local_size) != 0)
                ret = Array(self._gpu, shape=(m,))
                ret.job = f(tmp, ret)
                ret._keep = [tmp]

                if m == 1:
                    if keepdims:
                        shape = np.ones(shape=(np.asarray(self.shape).shape),
                                        dtype=int)
                        ret.reshape(shape)
                    return ret

                n = m
                tmp = ret
        else:
            return self._axis_reduction(spv_axis, axis, keepdims)

    def sum(self, axis: Union[int, Iterable[int]]=None,
            keepdims: bool = False,
            rebroadcast: bool = False) -> Array:
        """
        Calculate Sum of Elements

        Parameters
        ----------
        axis : int, optional
            Reduction axis
        keepdims : bool, optional
            When `True`, reduced dimensions are keeped with size one.
            Default is `False`.
        rebroadcast : bool, optional
            When `True`, keep shape by re-broadcasting after reduce.
            Default is `False`.

        Returns
        -------
        vulkpy.Array
            Summarized array
        """
        return self._reduce(self._sum,
                            self._sum_v1_3,
                            self._sum_axis,
                            self._sum_axis_rebroadcast,
                            axis,
                            keepdims,
                            rebroadcast)

    def prod(self, axis: Union[int, Iterable[int]]=None,
             keepdims: bool = False,
             rebroadcast: bool = False) -> Array:
        """
        Calculate Product of Elements

        Parameters
        ----------
        axis : int, optional
            Reduction axis
        keepdims : bool, optional
            When `True`, reduced dimensions are keeped with size one.
            Default is `False`.
        rebroadcast : bool, optional
            When `True`, keep shape by re-broadcasting after reduce.
            Default is `False`.

        Returns
        -------
        vulkpy.Array
            Producted array
        """
        return self._reduce(self._prod,
                            self._prod_v1_3,
                            self._prod_axis,
                            self._prod_axis_rebroadcast,
                            axis,
                            keepdims,
                            rebroadcast)

    def maximum(self, axis: Union[int, Iterable[int]]=None,
                keepdims: bool = False,
                rebroadcast: bool = False) -> Array:
        """
        Get Maximum Value

        Parameters
        ----------
        axis : int, optional
            Reduction axis
        keepdims : bool, optional
            When `True`, reduced dimensions are keeped with size one.
            Default is `False`.
        rebroadcast : bool, optional
            When `True`, keep shape by re-broadcasting after reduce.
            Default is `False`.

        Returns
        -------
        vulkpy.Array
            Maximum array
        """
        return self._reduce(self._maximum,
                            self._maximum_v1_3,
                            self._maximum_axis,
                            self._maximum_axis_rebroadcast,
                            axis,
                            keepdims,
                            rebroadcast)

    def minimum(self, axis: Union[int, Iterable[int]]=None,
                keepdims: bool = False,
                rebroadcast: bool = False) -> Array:
        """
        Get Minimum Value

        Parameters
        ----------
        axis : int, optional
            Reduction axis
        keepdims : bool, optional
            When `True`, reduced dimensions are keeped with size one.
            Default is `False`.
        rebroadcast : bool, optional
            When `True`, keep shape by re-broadcasting after reduce.
            Default is `False`.

        Returns
        -------
        vulkpy.Array
            Minimum array
        """
        return self._reduce(self._minimum,
                            self._minimum_v1_3,
                            self._minimum_axis,
                            self._minimum_axis_rebroadcast,
                            axis,
                            keepdims,
                            rebroadcast)

    def mean(self, axis: Union[int, Iterable[int]]=None,
             keepdims: bool = False,
             rebroadcast: bool = False) -> Array:
        """
        Calculate Mean Value

        Parameters
        ----------
        axis : int, optional
            Reduction axis
        keepdims : bool, optional
            When `True`, reduced dimensions are keeped with size one.
            Default is `False`.
        rebroadcast : bool, optional
            When `True`, keep shape by re-broadcasting after reduce.
            Default is `False`.

        Returns
        -------
        vulkpy.Array
            Mean array
        """
        n_before = self.buffer.size()

        ret = self.sum(axis, keepdims, rebroadcast)

        if rebroadcast:
            ret /= self.shape[axis]
        else:
            n_after = ret.buffer.size()
            ret *= (n_after/n_before)

        return ret

    def broadcast_to(self, shape: Iterable[int]) -> Array:
        """
        Broadcast to new shape

        Parameters
        ----------
        shape : iterable of ints
            Shape of broadcast target

        Returns
        -------
        vulkpy.Array
            Broadcasted array

        Raises
        ------
        ValueError
            If ``shape`` is not compatible.
        """
        shape = np.asarray(shape, dtype=int)
        if np.all(np.broadcast_shapes(self.shape, shape) != shape):
            raise ValueError(f"Cannot broadcast to {shape}")

        ret = Array(self._gpu, shape=shape)

        self_shape = self.shape
        dim_diff = len(shape) - len(self_shape)
        if dim_diff > 0:
            self_shape = np.concatenate((np.ones(shape=(dim_diff,)), self_shape),
                                        axis=0)

        shapeA = Shape(self._gpu, data=self_shape)
        shapeB = Shape(self._gpu, data=shape)

        ret.job = self._gpu._submit(self._broadcast, 64, 1, 1,
                                    [self, ret, shapeA, shapeB],
                                    DataShape(ret.buffer.size(), 1, 1),
                                    BroadcastParams(self.buffer.size(),
                                                    ret.buffer.size(),
                                                    shapeA.buffer.size()))
        ret._keep = [self, shapeA, shapeB]
        return ret

    def gather(self, indices: U32Array, axis: Optional[int] = None) -> Array:
        """
        Gather values of indices

        Parameters
        ----------
        indices : vulkpy.U32Array
            Indices
        axis : int, optional
            Axis of gather.
            If ``None`` (default), array is flattened beforehand.

        Returns
        -------
        vulkpy.Array
            Gathered array
        """
        size = indices.buffer.size()
        if axis is None:
            spv = self._gather
            local_size = (64, 1, 1)

            ret = Array(self._gpu, shape=indices.shape)

            d = DataShape(size, 1, 1)
            p = VectorParams(size)
        else:
            spv = self._gather_axis
            local_size = (1, 64, 1)

            shape = np.array(self.shape)
            prev_shape = shape[:axis]
            post_shape = shape[axis+1:]
            shape = np.concatenate((indices.shape, prev_shape, post_shape),
                                   axis=0)

            ret = Array(self._gpu, shape=shape)

            prev_prod = int(np.prod(prev_shape))
            axis_size = int(self.shape[axis])
            post_prod = int(np.prod(post_shape))
            d = DataShape(prev_prod, post_prod, size)
            p = AxisGatherParams(prev_prod, post_prod, axis_size, size)

        ret.job = self._gpu._submit(spv, *local_size, [self, indices, ret], d, p)
        ret._keep = [self, indices]
        return ret
