# pylint: disable=invalid-name, unused-argument
"""MNM sharding system utilities"""
import functools
from queue import PriorityQueue
from mnm._ffi.sharding._make import ShardOpAttrs
from mnm._ffi.op import GetOp
from mnm._lib import _register_func, relay
from mnm.distributed.sharding.shardspec import ReplicatedSpec, ShardSpec, TupleShardSpec
from mnm import distributed as dist
from tvm.relay import Call

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

def always_apply(call: relay.Call):
    """Always apply this pattern to expand op call"""
    return True

def expand_when(cond, priority=1):
    """Specify the priority and the condition when this expansion pattern should be used.

    Parameters
    ----------
    cond : function(call) -> bool
        A function answering this expansion pattern is eligible under particular conditions
        (e.g. with particular sharding specifications)
    """
    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_names"):
            raise ValueError("Must register expansion pattern first")
        for op_name in pyfunc.op_names:
            op = GetOp(op_name) if op_name != "_fallback" else "_fallback"
            if op not in _expansion_patterns:
                _expansion_patterns[op] = PriorityQueue()
            _expansion_patterns[op].put((-priority, cond, pyfunc))
        return pyfunc
    return decorator

def register_expansion_pattern(op_name):
    """Register an expansion pattern that converts a full-sized op into a partitioned-size op

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

def extract_shardOpCall(call):
    """Return some frequently-used object attributes as a tuple"""
    assert isinstance(call, relay.Call)
    return (call.op, call.args, call.attrs.shard_in, call.attrs.shard_out)

@_register_func("mnm.sharding._match_expansion_pattern")
def expand_shardOpCall(call: relay.Call):
    """Match an eligible expansion pattern and return expanded IR expr"""
    print("expand: ", call, call.attrs)
    patterns = _expansion_patterns[call.op if call.op in _expansion_patterns else "_fallback"]
    for pattern in patterns.queue:
        _, cond, irgen = pattern
        print(cond(call))
        if cond(call):
            break
    return irgen(call)

@expand_when(lambda call: isinstance(call.attrs.shard_in, ReplicatedSpec) and \
                          isinstance(call.attrs.shard_out, ShardSpec), priority=1)
@register_expansion_pattern("mnm.op.sharding._reshard")
def reshard_to_strided_slice(call: relay.Call):
    """_reshard -> strided_slice"""
    # GetOp("mnm.op.strided_slice")
    return call

@expand_when(always_apply, priority=0)
@register_expansion_pattern("mnm.op.sharding._reshard")
def reshard_mismatch(call: relay.Call):
    """_reshard -> <error>"""
    raise NotImplementedError("Unable to process the given sharding specifications")

@expand_when(always_apply)
@register_expansion_pattern(["mnm.op.add", "mnm.op.subtract"])
def add_or_sub(call: relay.Call):
    """add/sub -> (reshard) add/sub"""
    op, args, sin, sout = extract_shardOpCall(call)
    if not sin[0] == sin[1] == sout:
        args = [relay.Call(GetOp("mnm.op.sharding._reshard"),
                           [args[i]],
                           ShardOpAttrs(sin[i], sout)) for i in (0, 1)] + args[2:]
    return relay.Call(op, args)

@expand_when(always_apply)
@register_expansion_pattern("_fallback")
def fallback_reshard_to_replicated(call: relay.Call):
    """Gather partitioned tensors for op without matched patterns"""
    op, args, attrs = call.op, call.args, call.attrs
    if len(args) != 1 or \
            isinstance(attrs.shard_in, TupleShardSpec) or \
            isinstance(attrs.shard_out, TupleShardSpec):
        raise NotImplementedError("Currently coverting multiple args is not supported")
    new_attrs = ShardOpAttrs(attrs.shard_in, ReplicatedSpec())
    new_args = [relay.Call(GetOp("mnm.op.sharding._reshard"), args, new_attrs)]
    return relay.Call(op, new_args)

@_register_func("mnm.sharding._py_print")
def _py_print(obj):
    """Only for debugging"""
    print(obj)
