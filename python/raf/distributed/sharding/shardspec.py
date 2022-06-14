# pylint: disable=invalid-name, unused-argument
"""RAF sharding specifications and attributes."""
from raf._core.core_utils import register_node
from raf._ffi.sharding import _make
from raf._core.value import Value


@register_node("raf.sharding.BaseShardSpec")
class BaseShardSpec(Value):
    """Base type of Sharding Specifications"""

@register_node("raf.sharding.ShardSpec")
class ShardSpec(BaseShardSpec):
    """Sharding Specifications"""

    def __init__(self, ranks, phy_shape, subgroup_shape):
        self.__init_handle_by_constructor__(_make.ShardSpec, ranks, phy_shape, subgroup_shape)


@register_node("raf.sharding.UnsetShardSpec")
class UnsetShardSpec(BaseShardSpec):
    """Placeholder of Sharding Specifications"""

    def __init__(self):
        self.__init_handle_by_constructor__(_make.UnsetShardSpec)