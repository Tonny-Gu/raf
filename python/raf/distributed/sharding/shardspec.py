# pylint: disable=invalid-name, unused-argument
"""RAF sharding specifications and attributes."""
from raf._core.core_utils import register_node
from raf._ffi.sharding import _make
from raf._lib import Object
from raf._core.value import Value


@register_node("raf.sharding.BaseShardSpec")
class BaseShardSpec(Value):
    """Base type of Sharding Specifications"""


@register_node("raf.sharding.ReplicatedSpec")
class ReplicatedSpec(BaseShardSpec):
    """Annotation denoting every rank has a full copy of this tensor"""

    def __init__(self, immutable=False):
        self.__init_handle_by_constructor__(_make.ReplicatedSpec, immutable)


@register_node("raf.sharding.TupleShardSpec")
class TupleShardSpec(BaseShardSpec):
    """Denote a OpCall with multiple input or output shards"""

    def __init__(self, tuple_elem, immutable=False):
        assert isinstance(tuple_elem, list)
        self.__init_handle_by_constructor__(_make.TupleShardSpec, immutable, tuple_elem)

    def __getitem__(self, index: int):
        return self.tuple_elem[index]

    def __len__(self):
        return len(self.tuple_elem)


@register_node("raf.sharding.ShardSpec")
class ShardSpec(BaseShardSpec):
    """Annotation of Sharding Specifications"""

    def __init__(self, ranks, real_shape, replicas, immutable=False):
        self.__init_handle_by_constructor__(_make.ShardSpec, immutable, ranks, real_shape, replicas)
