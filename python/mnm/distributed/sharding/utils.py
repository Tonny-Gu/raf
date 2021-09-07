"""MNM sharding system utilities"""
# pylint: disable=invalid-name, unused-argument
import functools
from queue import PriorityQueue
from mnm._core.core_utils import str2dev
from mnm._ffi.device import Device
from mnm._ffi.op import GetOp
from mnm._lib import _register_func, relay
from mnm.testing.common import get_device_list
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
    
def get_dist_device_list(dev_type="cuda"):
    """Return all available devices in the cluster as a list of Device Objects.

    Parameters
    ----------
    dev_type : str
        The device type expected to use

    Returns
    -------
    ret: list
        List of Device Objects
    """
    if dev_type not in get_device_list():
        raise RuntimeError("Non-existing Device Type: " + dev_type)
    dev_type_id = str2dev(dev_type).device_type
    dctx = dist.get_context()
    dev_array = [Device(dev_type_id, i) for i in range(dctx.size*16)]
    #TODO: size*16 is only for testing
    return dev_array

_expanding_rules = {}

def always_apply(op, args, attrs):
    """Always apply this rule to expand op call"""
    return True

def expand_when(cond, priority=1):
    """Specify the priority and the condition when this expanding rule should be used.

    Parameters
    ----------
    cond : function(op, args, attrs) -> bool
        A function answering this expanding rule is eligible under particular conditions
        (e.g. with particular sharding specifications)
    """
    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_name"):
            raise ValueError("Must register expanding rule first")
        op = GetOp(pyfunc.op_name) if pyfunc.op_name is not "_fallback" else "_fallback"
        if pyfunc.op_name not in _expanding_rules:
            _expanding_rules[op] = PriorityQueue()
        _expanding_rules[op].put((priority, cond, pyfunc))
        return pyfunc
    return decorator

def register_expanding_rule(op_name):
    """Register an expanding rule that converts a full-sized op into a partitioned-size op"""
    def decorator(pyfunc):
        @functools.wraps(pyfunc)
        def new_pyfunc(op, args, attrs):
            return pyfunc(op, args, attrs)
        setattr(new_pyfunc, "op_name", op_name)
        return new_pyfunc
    return decorator


@_register_func("mnm.sharding._match_expanding_rule")
def expand_shardOpCall(op, args, attrs):
    """Match an eligible expanding rule and return expanded IR expr"""
    rules = _expanding_rules[op if op in _expanding_rules else "_fallback"]
    for rule in rules.queue:
        _, cond, irgen = rule
        if cond(op, args, attrs):
            break
    return irgen(op, args, attrs)

def is_elemwise_op_with_same_shardspec(op, args, attrs):
    """Check op is element wise while input's shardspec is the same as output's"""
    return pattern_map[op.get_attr("TOpPattern")] == "kElemWise" \
            and attrs.shard_in == attrs.shard_out

@expand_when(is_elemwise_op_with_same_shardspec, priority=1)
@register_expanding_rule("_fallback")
def fallback_elemwise_op(op, args, attrs):
    """Convert elemwise op"""
    return relay.Call(op, args)

@expand_when(always_apply, priority=0)
@register_expanding_rule("_fallback")
def fallback_reshard_to_replicated(op, args, attrs):
    """Gather partitioned tensors for op without matched rules"""
    #TODO: add reshard
    return relay.Call(op, args)
