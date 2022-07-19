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
from raf.distributed.sharding.utils import make_replicated_spec
from raf._core.value import Value
from raf import distributed as dist
from raf.ir.anf_builder import ANFBuilder
from tvm.relay import Call, Expr
from tvm.ir import Op

from .expandrule import ShardInfo, all_satisfied, always_apply, expand_opcall, is_same_spec, is_sharded
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

    # Step 1: Propagate ShardSpec
    filled_sin = []
    for i in range(len(s.sin)):
        if isinstance(s.sin[i], UnsetShardSpec):
            if isinstance(s.args[i], relay.Call) and hasattr(s.args[i].attrs, "sin"):
                # if isinstance(s.args[i].\attrs, ShardOpCallAttrs):
                prev_sinfo = ShardInfo(s.args[i])
                filled_sin.append(prev_sinfo.sout[0])
            else:
                ndim = len(s.args[0].checked_type.concrete_shape)
                filled_sin.append(make_replicated_spec(ndim))

        else:
            filled_sin.append(s.sin[i])
    
    filled_attrs = ShardOpCallAttrs(filled_sin, s.sout)
    filled_call = relay.Call(s.op, s.args, filled_attrs)
    filled_s = ShardInfo(filled_call)

    # Step 2: Query InferHint
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(filled_s):
            break
    else:
        raise ValueError("Failed to match an InferHint")
    possible_calls = [relay.Call(s.op, s.args, a) for a in irgen(filled_s)]
    print(possible_calls)

    # Step 3: Check attr is accepted by a Expansion Rule
    possible_calls = list(filter(lambda x: expand_opcall(x) != None, possible_calls))

    # Step 4: Select a OpCall with full ShardSpecs
    # TODO: should use graph searching algorithm with cost map here. For now, always select the first solution.
    sol_call = possible_calls[0]
    sol_s = ShardInfo(sol_call)

    # Step 5: Insert Reshard OpCall
    resharded_args = []
    for i in range(len(filled_s.sin)):
        if is_same_spec(filled_s.sin[i], sol_s.sin[i]):
            resharded_args.append(sol_s.args[i])
        else:
            resharded_args.append(relay.Call(
                GetOp("raf.op._reshard"),
                [sol_s.args[i]],
                ShardOpCallAttrs([filled_s.sin[i]], [sol_s.sin[i]])))

    return relay.Call(sol_s.op, resharded_args, sol_s.attrs)

def is_unset(s: BaseShardSpec):
    return isinstance(s, UnsetShardSpec)

@try_when(always_apply)
@register_infer_hint(["raf.op.add", "raf.op.subtract"])
def element_wise_op_with_2in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sin[1], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [
        ShardOpCallAttrs([e, e], [e]) for e in specs
    ]

@try_when(always_apply)
@register_infer_hint(["raf.op.relu"])
def element_wise_op_with_1in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [
        ShardOpCallAttrs([e], [e]) for e in specs
    ]
