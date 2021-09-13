"""MNM sharding system"""
from mnm._ffi.sharding._make import ShardOpAttrs
from .shardspec import BaseShardSpec, ReplicatedSpec, TupleShardSpec, ShardSpec
from .utils import get_dist_devices, expand_when, always_apply, register_expansion_pattern
