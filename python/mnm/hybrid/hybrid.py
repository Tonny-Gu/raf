import ast
import inspect
import numpy as np

from typing import Callable, Dict

from .._core.base import set_module
from .._core.context import cpu
from .._core.ndarray import ndarray as NDArray
from .._core.ir import Module, ConstantExpr
from .._core.value import IntValue, FloatValue, BoolValue, TensorValue
from .._core.bound_expr import BoundExpr
from .._core.executor import Interpreter
from .._ffi._tvm import relay
from .cfg import ast2cfg
from .ir_builder import build_ir
from .sanity_check import sanity_check
from .to_builder import to_builder
from .to_relay import cfg2relay
from .utils import get_func_name

FUNC_TAB: Dict[Callable, Callable] = {}
FUNC_VAR: Dict[Callable, relay.GlobalVar] = {}
MNM_MODULE = Module.GLOBAL


def find_invoker_name(namespace) -> str:
    name = "__ir_builder_invoker"
    if name not in namespace:
        return name
    i = 0
    while True:
        new_name = name + "$" + str(i)
        if new_name not in namespace:
            return new_name
        i += 1


ARG_MAKERS = {
    int: ConstantExpr,
    float: ConstantExpr,
    bool: ConstantExpr,
    np.ndarray: ConstantExpr,
    NDArray: None,  # TODO
}


def _make_argument(a):
    if type(a) not in ARG_MAKERS:
        raise NotImplementedError
    return ARG_MAKERS[type(a)](a)


def _unwrap(a):
    if isinstance(a, IntValue):
        return a.data
    if isinstance(a, FloatValue):
        return a.data
    if isinstance(a, BoolValue):
        return bool(a.data)
    if isinstance(a, BoundExpr):
        return NDArray(a)
    if isinstance(a, list):
        return [_unwrap(item) for item in a]
    if isinstance(a, tuple):
        return tuple(_unwrap(item) for item in a)
    raise NotImplementedError


def pyfunc2relay(pyfunc, entry: relay.GlobalVar):
    mem = dict(inspect.getmembers(pyfunc))
    # getting AST
    node = ast.parse(inspect.getsource(pyfunc))
    ast.increment_lineno(node, mem['__code__'].co_firstlineno - 1)
    # AST -> IR builder
    node = sanity_check(node)
    invoker_name = find_invoker_name(mem['__globals__'])
    node, local_names = to_builder(node, pyfunc, invoker_name)
    compiled = compile(node, filename="<string>", mode='exec')
    # IR builder -> AST
    # TODO(@junrushao1994): deal with nonlocals
    exec(compiled, mem['__globals__'])
    node = build_ir(mem['__globals__'][invoker_name], debug=False)
    # AST -> CFG
    cfg = ast2cfg(node)
    # CFG -> Relay
    local_names = list(local_names)
    hybrid_module = cfg2relay(cfg, pyfunc, local_names, entry)
    # build relay module
    for global_var, func in hybrid_module.items():
        MNM_MODULE[global_var] = func

    def call(*args):
        code = relay.Call(op=entry, args=[_make_argument(arg) for arg in args])
        result = Interpreter.GLOBAL(code)
        return _unwrap(result)

    return call


@set_module("mnm")
def hybrid(python=False):

    def hybrid_no_python(pyfunc):
        global FUNC_TAB
        global FUNC_VAR
        func_name = get_func_name(pyfunc)
        sig = inspect.signature(pyfunc)
        if pyfunc not in FUNC_TAB:
            FUNC_TAB[pyfunc] = None
            FUNC_VAR[pyfunc] = relay.GlobalVar(func_name)

        def transformed(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            pos_args = list(bound.arguments.values())

            func = FUNC_TAB[pyfunc]
            if func is None:
                func = pyfunc2relay(pyfunc, FUNC_VAR[pyfunc])
                FUNC_TAB[pyfunc] = func
            return func(*pos_args)

        return transformed

    if callable(python):
        return hybrid_no_python(python)
    assert not python
    return hybrid_no_python