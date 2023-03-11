from __future__ import annotations
from typing import Tuple, Union
from typing_extensions import Protocol

import numpy as np


KeyType = Union[int, np.ndarray, slice]
ValueType = Union[int, float, np.ndarray]

class Resource:
    pass

class ArrayProtocol(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def array(self) -> np.ndarray: ...

    def wait(self): ...
