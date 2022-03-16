# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
import raf
from raf._core.core_utils import str2dev
from raf._core.executor import interpret
from raf.distributed.sharding import (
    ShardSpec,
    ReplicatedSpec,
    TupleShardSpec,
    BaseShardSpec,
    ShardOpCallAttrs,
)
from raf._ffi.pass_ import SetShardOpCallAttrs, ToGraphNormalForm, ExpandShardOpCall, InferType
from raf._ffi.device import Device
from raf._lib import relay
from raf.distributed.sharding.utils import get_dist_devices
from raf.testing import randn
from raf.hybrid.hybrid import _make_argument, _unwrap
from raf import distributed as dist
from tvm.relay.analysis.analysis import post_order_visit


def test_ShardOpCallAttrs():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            return z

    model = Model()
    # m_x, _ = randn((128, 128))
    # m_y, _ = randn((128, 128))
    m_x = raf.array([1, 2, 3, 4])
    m_y = raf.array([0, 0, 0, 0])
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    def get_global_device_list(dev_type="cuda"):
        dev_type_id = str2dev(dev_type).device_type
        dctx = dist.get_context()
        local_id = 6
        local_id -= 1
        dev_array = (
            [Device(dev_type_id, i) for i in range(1, local_id)]
            + [dctx.local_device]
            + [Device(dev_type_id, i) for i in range(local_id, 16)]
        )
        return dev_array

    # devices = get_global_device_list()
    devices = get_dist_devices()
    attrs = ShardOpCallAttrs(
        TupleShardSpec([ReplicatedSpec(), ReplicatedSpec()]), ShardSpec(devices, [4], [1])
    )
    # a = raf._reshard_r2s(m_x, ShardSpec(devices, [4, 4], [1, 2]))
    # print(a)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    attrs_map = {call_list[0]: attrs}

    mod0 = SetShardOpCallAttrs(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = ExpandShardOpCall()(mod1)
    # print(raf._ffi.ir.AsText(mod2))
    call = relay.Call(op=mod2["main"], args=[_make_argument(x) for x in (m_x, m_y)])
    result = _unwrap(interpret(call, mod2))
    print(result)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_ShardOpCallAttrs()
