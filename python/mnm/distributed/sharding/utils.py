# pylint: disable=invalid-name, unused-argument
"""MNM sharding system utilities"""
import functools
from queue import PriorityQueue
from mnm._ffi.sharding._make import ShardOpAttrs
from mnm._ffi.op import GetOp
from mnm._lib import _register_func, relay
from mnm.distributed.sharding.shardspec import ReplicatedSpec, ShardSpec, TupleShardSpec
from mnm import distributed as dist

pattern_map = {
    0: "kElemWise",
    1: "kBroadcast",
    2: "kInjective",
    3: "kCommReduce",
    4: "kOutEWiseFusable",
    7: "kTuple",
    8: "kOpaque",
}
#TODO: this pattern map is replicated mulitple times in source code
    
def get_dist_devices():
    """Return all available devices in the cluster as a list of Device Objects.

    Returns
    -------
    ret: list
        List of Device Objects
    """
    return dist.get_context().dist_devices

_expansion_patterns = {}

def always_apply(op, args, attrs):
    """Always apply this pattern to expand op call"""
    return True

def expand_when(cond, priority=1):
    """Specify the priority and the condition when this expansion pattern should be used.

    Parameters
    ----------
    cond : function(op, args, attrs) -> bool
        A function answering this expansion pattern is eligible under particular conditions
        (e.g. with particular sharding specifications)
    """
    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_name"):
            raise ValueError("Must register expansion pattern first")
        op = GetOp(pyfunc.op_name) if pyfunc.op_name is not "_fallback" else "_fallback"
        if pyfunc.op_name not in _expansion_patterns:
            _expansion_patterns[op] = PriorityQueue()
        _expansion_patterns[op].put((priority, cond, pyfunc))
        return pyfunc
    return decorator

def register_expansion_pattern(op_name):
    """Register an expansion pattern that converts a full-sized op into a partitioned-size op"""
    def decorator(pyfunc):
        @functools.wraps(pyfunc)
        def new_pyfunc(op, args, attrs):
            return pyfunc(op, args, attrs)
        setattr(new_pyfunc, "op_name", op_name)
        return new_pyfunc
    return decorator


@_register_func("mnm.sharding._match_expansion_pattern")
def expand_shardOpCall(op, args, attrs):
    """Match an eligible expansion pattern and return expanded IR expr"""
    patterns = _expansion_patterns[op if op in _expansion_patterns else "_fallback"]
    for pattern in patterns.queue:
        _, cond, irgen = pattern
        if cond(op, args, attrs):
            break
    return irgen(op, args, attrs)

@expand_when(lambda _, __, attrs: isinstance(attrs.shard_in, ReplicatedSpec) and \
                                  isinstance(attrs.shard_out, ShardSpec), priority=1)
@register_expansion_pattern("mnm.sharding._reshard")
def reshard_to_strided_slice(op, args, attrs):
    """_reshard -> strided_slice"""
    return relay.Call(op, args)

def reshard_mismatch(op, args, attrs):
    """_reshard -> <error>"""
    raise NotImplementedError("Unable to process the given sharding specifications")

def is_elemwise_op(op, args, attrs):
    """Check op is element wise"""
    return pattern_map[op.get_attr("TOpPattern")] == "kElemWise"

@expand_when(is_elemwise_op, priority=1)
@register_expansion_pattern("_fallback")
def fallback_elemwise_op(op, args, attrs):
    """Convert elemwise op"""
    if attrs.shard_in != attrs.shard_out:
        if len(args) != 1 or \
                isinstance(attrs.shard_in, TupleShardSpec) or \
                isinstance(attrs.shard_out, TupleShardSpec):
            raise NotImplementedError("Currently tuple is not supported")
        args = relay.Call(GetOp("mnm.sharding._reshard"), args, attrs)
    return relay.Call(op, args)

@expand_when(always_apply, priority=0)
@register_expansion_pattern("_fallback")
def fallback_reshard_to_replicated(op, args, attrs):
    """Gather partitioned tensors for op without matched patterns"""
    if len(args) != 1 or \
            isinstance(attrs.shard_in, TupleShardSpec) or \
            isinstance(attrs.shard_out, TupleShardSpec):
        raise NotImplementedError("Currently tuple is not supported")
    new_attrs = ShardOpAttrs(attrs.shard_in, ReplicatedSpec())
    new_args = [relay.Call(GetOp("mnm.sharding._reshard"), args, new_attrs)]
    return relay.Call(op, new_args)

@_register_func("mnm.sharding._py_print")
def _py_print(obj):
    """Only for debugging"""
    print(obj)
