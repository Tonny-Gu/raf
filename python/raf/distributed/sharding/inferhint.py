# pylint: disable=invalid-name, unused-argument
"""Implementaion of Infer Hints"""
from ctypes import Union
import functools
import numpy as np
import raf
import tvm
from queue import PriorityQueue
from typing import Callable, List, Tuple

from raf._ffi.sharding._make import ShardOpCallAttrs
from raf._ffi.op import GetOp
from raf._lib import _register_func, relay
from raf.distributed.sharding.shardspec import BaseShardSpec, ShardSpec, UnsetShardSpec
from raf._core.value import Value
from raf import distributed as dist
from raf.ir.anf_builder import ANFBuilder
from tvm.relay import Call, Expr
from tvm.ir import Op

from .expandrule import ShardInfo, all_satisfied, always_apply, is_sharded
from .expandrule import register_expansion_rule as register_infer_hint

def try_when(cond: Callable[[ShardInfo], bool], priority=1):
    if not hasattr(try_when, "counter"):
        try_when.counter = 0
    if not hasattr(try_when, "rules"):
        try_when.rules = {}

    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_names"):
            raise ValueError("Must register infer hint first")
        for op_name in pyfunc.op_names:
            op = GetOp(op_name)
            if op not in try_when.rules:
                try_when.rules[op] = PriorityQueue()
            try_when.rules[op].put((-priority, try_when.counter, cond, pyfunc))
            try_when.counter += 1
        return pyfunc

    return decorator

@_register_func("raf.sharding._infer_shardspec")
def infer_shardspec(call: relay.Call):
    rules = try_when.rules[call.op]
    s = ShardInfo(call)
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(s):
            break
    return irgen(s)

def is_unset(s: BaseShardSpec):
    return isinstance(s, UnsetShardSpec)

@try_when(always_apply)
@register_infer_hint(["raf.op.add", "raf.op.subtract"])
def element_wise_op_with_2in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    print("Hello")
    specs = []
    for e in (s.sin[0], s.sin[1], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [
        ShardOpCallAttrs([e, e], [e]) for e in specs
    ]
