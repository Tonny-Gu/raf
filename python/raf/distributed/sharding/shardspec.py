# pylint: disable=invalid-name, unused-argument
"""RAF sharding specifications and attributes."""
from raf._core.core_utils import register_node
from raf._ffi.sharding import _make
from raf._lib import Object
from raf._core.value import Value


@register_node("raf.sharding.BaseSpecValue")
class BaseSpecValue(Value):
    """Base type of Sharding Specifications"""


@register_node("raf.sharding.ReplicatedSpecValue")
class ReplicatedSpecValue(BaseSpecValue):
    """Annotation denoting every rank has a full copy of this tensor"""

    def __init__(self, immutable=False):
        self.__init_handle_by_constructor__(_make.ReplicatedSpecValue, immutable)


@register_node("raf.sharding.TupleSpecValue")
class TupleSpecValue(BaseSpecValue):
    """Denote a OpCall with multiple input or output shards"""

    def __init__(self, tuple_elem, immutable=False):
        assert isinstance(tuple_elem, list)
        self.__init_handle_by_constructor__(_make.TupleSpecValue, immutable, tuple_elem)

    def __getitem__(self, index: int):
        return self.tuple_elem[index]

    def __len__(self):
        return len(self.tuple_elem)


@register_node("raf.sharding.ShardSpecValue")
class ShardSpecValue(BaseSpecValue):
    """Annotation of Sharding Specifications"""

    def __init__(self, ranks, phy_shape, replicas, immutable=False):
        self.__init_handle_by_constructor__(_make.ShardSpecValue, immutable, ranks, phy_shape, replicas)
