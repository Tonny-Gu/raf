# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
from functools import reduce
import operator

from tvm import relay

from . import cuda
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .._lib import strategy
from .._lib import random

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member

@register_compute("mnm.op.tvm._reshard_r2s")
def compute_reshard_r2s(attr, inputs, output_type):
    # pylint: disable=unused-argument
    x = inputs[0]
    return [_topi.strided_slice(x, [0, 0], [1, 1])]

_reg.register_injective_schedule("mnm.op.tvm._reshard_r2s")