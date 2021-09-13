# pylint: disable=invalid-name, unused-argument
"""MNM sharding specifications and attributes."""
from mnm._core.core_utils import register_node
from mnm._ffi.sharding import _make
from mnm._lib import Object

@register_node("mnm.sharding.BaseShardSpec")
class BaseShardSpec(Object):
    """Base type of Sharding Specifications"""

@register_node("mnm.sharding.ReplicatedSpec")
class ReplicatedSpec(BaseShardSpec):
    """Annotation denoting every node has a copy of the data"""
    def __init__(self, immutable=False):
        self.__init_handle_by_constructor__(_make.ReplicatedSpec, immutable)

@register_node("mnm.sharding.TupleShardSpec")
class TupleShardSpec(BaseShardSpec):
    """Annotation of a tuple that will usually be used
       when having multiple input or output tensors"""
    def __init__(self, tuple_elem, immutable=False):
        assert isinstance(tuple_elem, list)
        self.__init_handle_by_constructor__(_make.TupleShardSpec, immutable, tuple_elem)

    def __getitem__(self, index: int):
        return self.tuple_elem[index]

    def __len__(self):
        return len(self.tuple_elem)

@register_node("mnm.sharding.ShardSpec")
class ShardSpec(BaseShardSpec):
    """Generic annotation of Sharding Specifications"""
    def __init__(self,
                 assigned_devices,
                 num_devices_on_dim,
                 num_replicas_on_dim,
                 immutable=False):
        self.__init_handle_by_constructor__(_make.ShardSpec,
                                            immutable,
                                            assigned_devices,
                                            num_devices_on_dim,
                                            num_replicas_on_dim)
