# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
import mnm
from mnm._core.core_utils import str2dev
from mnm.distributed.sharding import ShardSpec, ReplicatedSpec, TupleShardSpec, BaseShardSpec, ShardOpAttrs
from mnm._ffi.pass_ import SetShardOpAttrs, ToGraphNormalForm, ExpandShardOpCall, InferType
from mnm._ffi.device import Device
from mnm._lib import relay
from mnm.testing import randn
from mnm import distributed as dist
from tvm.relay.analysis.analysis import post_order_visit

def test_shardOpAttrs():

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            return z

    model = Model()
    m_x, _ = randn((128, 128))
    m_y, _ = randn((128, 128))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    def get_global_device_list(dev_type="cuda"):
        dev_type_id = str2dev(dev_type).device_type
        dctx = dist.get_context()
        local_id = 6
        local_id -= 1
        dev_array = [Device(dev_type_id, i) for i in range(1, local_id)] + \
                    [dctx.local_device] + [Device(dev_type_id, i) for i in range(local_id, 16)]
        return dev_array
    devices = get_global_device_list()
    attrs = ShardOpAttrs(TupleShardSpec([ReplicatedSpec(), ReplicatedSpec()]),
                         ShardSpec(devices, [4, 4], [2, 2]))
    mnm._reshard_r2s(m_x, ShardSpec(devices, [4, 4], [2, 2]))
    return
    call_list = []
    post_order_visit(mod_before["main"].body,
                     lambda op: call_list.append(op) if isinstance(op, relay.Call) else None)
    attrs_map = {call_list[0] : attrs}

    mod = SetShardOpAttrs(attrs_map)(mod_before)
    mod = ToGraphNormalForm()(mod)
    print(mnm._ffi.ir.AsText(mod))
    mod1 = ExpandShardOpCall()(mod)
    print(mnm._ffi.ir.AsText(mod1))

if __name__ == "__main__":
    # pytest.main([__file__])
    test_shardOpAttrs()
