# pylint: disable=invalid-name, unused-argument
"""Implementation of Expansion Rules"""
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

pattern_map = {
    0: "kElemWise",
    1: "kBroadcast",
    2: "kInjective",
    3: "kCommReduce",
    4: "kOutEWiseFusable",
    7: "kTuple",
    8: "kOpaque",
}
# TODO: this pattern map is replicated multiple times in source code

class ShardInfo:
    call: relay.Call
    op: Op
    args: List[Expr]
    sin: List[BaseShardSpec]
    sout: List[BaseShardSpec]
    
    def __init__(self, call: relay.Call):
        assert isinstance(call, relay.Call)
        self.call = call
        self.op = call.op
        self.args = call.args
        self.sin = call.attrs.sin
        self.sout = call.attrs.sout

def all_satisfied(conds: List[Callable[[ShardInfo], bool]]):
    def func(s: ShardInfo):
        for c in conds:
            if not c(s):
                return False
        return True
    return func

def is_sharded(s: BaseShardSpec):
    return isinstance(s, ShardSpec)

def is_replicated(s: BaseShardSpec):
    if not isinstance(s, ShardSpec):
        return False
    return s.nshard == 1

def no_subgroup(s: BaseShardSpec):
    if not isinstance(s, ShardSpec):
        return False
    return s.ngroup == 1

def always_apply(s: ShardInfo):
    """Always apply this rule to expand op call"""
    return True

def expand_when(cond: Callable[[ShardInfo], bool], priority=1):
    """Specify the priority and the condition when this expansion rule should be used.

    Parameters
    ----------
    cond : function(call) -> bool
        A function answering this expansion rule is eligible under particular conditions
        (e.g. with particular sharding specifications)
    """
    if not hasattr(expand_when, "counter"):
        expand_when.counter = 0
    if not hasattr(expand_when, "rules"):
        expand_when.rules = {}

    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_names"):
            raise ValueError("Must register expansion rule first")
        for op_name in pyfunc.op_names:
            op = GetOp(op_name)
            if op not in expand_when.rules:
                expand_when.rules[op] = PriorityQueue()
            expand_when.rules[op].put((-priority, expand_when.counter, cond, pyfunc))
            expand_when.counter += 1
        return pyfunc

    return decorator


def register_expansion_rule(op_name):
    """Register an expansion rule that converts a full-sized op into a partitioned-size op

    Parameters
    ----------
    op_name: str or List[str]
        Name of op to register
    """
    op_names = [op_name] if isinstance(op_name, str) else op_name
    assert isinstance(op_names, list)

    def decorator(pyfunc):
        @functools.wraps(pyfunc)
        def new_pyfunc(call: relay.Call):
            return pyfunc(call)

        setattr(new_pyfunc, "op_names", op_names)
        return new_pyfunc

    return decorator

@_register_func("raf.sharding._match_expansion_rule")
def expand_opcall(call: relay.Call):
    """Match an eligible expansion rule and return expanded IR expr"""
    rules = expand_when.rules[call.op]
    s = ShardInfo(call)
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(s):
            break
    return irgen(s)

@expand_when(
    all_satisfied([
        lambda s: is_replicated(s.sin[0]),
        lambda s: is_sharded(s.sout[0])
    ]),
    priority=1,
)
@register_expansion_rule("raf.op._reshard")
def reshard_replicated_to_sharded(s: ShardInfo):
    """_reshard -> _reshard_r2s (strided_slice)"""
    spec = Value.as_const_expr(s.sout)
    return relay.Call(GetOp("raf.op._reshard_r2s"), [s.args[0], spec])

@expand_when(
    all_satisfied([
        lambda s: is_sharded(s.sout[1]),
        lambda s: is_replicated(s.sin[0]),
    ]),
    priority=1,
)
@register_expansion_rule("raf.op._reshard")
def reshard_sharded_to_replicated(s: ShardInfo):
    """_reshard -> _reshard_s2r (allgather)"""
    return relay.Call(GetOp("raf.op._reshard_s2r"), [s.args[0], s.sin])

# @expand_when(always_apply, priority=0)
# @register_expansion_rule("raf.op._reshard")
# def reshard_mismatch(s: ShardInfo):
#     """_reshard -> <error>"""
#     raise NotImplementedError("Unable to process the given sharding specifications")


@expand_when(lambda s: s.sin[0] == s.sin[1] == s.sout[0])
@register_expansion_rule(["raf.op.add", "raf.op.subtract"])
def add_or_sub(s: ShardInfo):
    """add/sub -> add/sub"""
    return relay.Call(s.op, s.args)

@expand_when(all_satisfied([
    lambda s: is_sharded(s.sin[0]) and is_sharded(s.sin[1]),
    lambda s: no_subgroup(s.sin[0]) and no_subgroup(s.sin[1]),
    lambda s: is_replicated(s.sout[0]),
    lambda s: s.sin[0].logic_shape[1] == s.sin[1].logic_shape[0]
]))
@register_expansion_rule(["raf.op.matmul"])
def matmul_algor1(s: ShardInfo):
    y_1 = relay.Call(s.op, s.args)
    y_2 = tvm.relay.Tuple([y_1])
    return relay.Call(GetOp("raf.op._allreduce"), [y_2, raf.ir.const("sum"), raf.ir.const(None)])

# @expand_when(always_apply)
# @register_expansion_rule("_fallback")
# def fallback_reshard_to_replicated(s: ShardInfo):
#     """Gather partitioned tensors for op without matched rules"""
#     op, args, attrs = call.op, call.args, call.attrs
#     if (
#         len(args) != 1
#         or isinstance(attrs.shard_in, TupleSpec)
#         or isinstance(attrs.shard_out, TupleSpec)
#     ):
#         raise NotImplementedError("Currently coverting multiple args is not supported")
#     new_attrs = ShardOpCallAttrs(attrs.shard_in, MirroredSpec())
#     new_args = [relay.Call(GetOp("raf.op._reshard"), args, new_attrs)]
#     return relay.Call(op, new_args)
