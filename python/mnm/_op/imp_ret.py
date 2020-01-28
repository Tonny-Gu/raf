from mnm._core.ndarray import ndarray
from mnm._core.value import BoolValue, FloatValue, IntValue, StringValue
from mnm._lib import Array, relay


def Any(x):  # pylint: disable=invalid-name
    if isinstance(x, (IntValue, FloatValue, StringValue)):
        return x.data
    if isinstance(x, BoolValue):
        return bool(x.data)
    if isinstance(x, relay.Var):
        return ndarray(x)
    if isinstance(x, tuple):
        return tuple(map(Any, x))
    if isinstance(x, (list, Array)):
        return list(map(Any, x))
    raise NotImplementedError(type(x))


def ArrayLike(x):  # pylint: disable=invalid-name
    if isinstance(x, (IntValue, FloatValue, StringValue)):
        return x.data
    if isinstance(x, BoolValue):
        return bool(x.data)
    if isinstance(x, relay.Var):
        return ndarray(x)
    raise NotImplementedError(type(x))


def TupleTensor(x):  # pylint: disable=invalid-name
    assert isinstance(x, Array)
    return [Tensor(y) for y in x]


def Tensor(x):  # pylint: disable=invalid-name
    if isinstance(x, relay.Var):
        return ndarray(x)
    raise NotImplementedError(type(x))