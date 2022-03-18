"""RAF sharding system"""
from raf._ffi.sharding._make import ShardOpCallAttrs
from .shardspec import BaseShardSpec, ReplicatedSpec, TupleShardSpec, ShardSpec
from .utils import (
    expand_when,
    always_apply,
    register_expansion_pattern,
    extract_shardOpCall,
)
