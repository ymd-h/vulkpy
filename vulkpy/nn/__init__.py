"""
Neural Network Module (:mod:`vulkpy.nn`)
========================================

Examples
--------
>>> import vulkpy as vk
>>> from vulkpy import nn
>>> gpu = vk.GPU()
>>> x = vk.Array(gpu, data=[ ... ]) # Features
>>> y = vk.Array(gpu, data=[ ... ]) # Labels

Create Optimizer and Model

>>> opt = nn.Adam(gpu, lr=1e-4)
>>> net = nn.Sequence(
...   [
...     nn.Dense(gpu, 3, 32, w_opt=opt, b_opt=opt),
...     nn.ReLU(),
...     nn.Dense(gpu, 32, 4, w_opt=opt, b_opt=opt),
...     nn.Softmax(),
...   ],
...   nn.CrossEntropy()
... )

Training Model

>>> pred_y, loss = net.train(x, y)

Predict with Model

>>> pred_y = net.predict(x)
"""

from .initializers import Constant, HeNormal
from .optimizers import SGD, Adam
from .layers import Dense, ReLU, Sigmoid, Softmax
from .losses import (
    CrossEntropyLoss,
    SoftmaxCrossEntropyLoss,
    MSELoss,
    HuberLoss,
)
from .models import Sequence
