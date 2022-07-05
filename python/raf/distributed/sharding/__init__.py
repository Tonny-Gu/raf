"""RAF sharding system"""

from raf._ffi.sharding._make import ShardOpCallAttrs
from .shardspec import BaseShardSpec, ShardSpec, UnsetShardSpec
from .utils import make_replicated_spec, make_shard_spec
from .expandrule import expand_opcall, register_expansion_rule, expand_when