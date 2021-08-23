# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import mnm
from mnm._core.core_utils import str2dev
from mnm._ffi.sharding._make import ShardSpec
from mnm._ffi.pass_ import AutoDiff, GradientInputSelection, InferType, SetShardOpAttrs
from mnm._ffi.device import Device
from mnm._lib import tvm
from mnm._lib import relay
from mnm.testing import randn, run_infer_type
from mnm._core.module import IRModule
from mnm.testing.common import get_device_list
from mnm import distributed as dist
from mnm.utils.visualizer import draw_dataflow_graph
from tvm.relay.analysis.analysis import post_order_visit

def get_dist_device_array(dev_type="cuda"):
    if dev_type not in get_device_list():
        raise RuntimeError("Non-existing Device Type: " + dev_type)
    dev_type_id = str2dev(dev_type).device_type
    dctx = dist.get_context()
    dev_array = [Device(dev_type_id, i) for i in range(dctx.size*16)]
    return dev_array

def test_shardOpAttrs():

    class Model(mnm.Model):
        def build(self):
            self.w, _ = randn((128, 128))
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.w)
            z = mnm.relu(y)
            return z

    model = Model()
    m_x, _ = randn((128, 128))
    m_x.requires_grad = True
    record = model._internal(m_x)
    mod_before = record.mod
    # mod_before = InferType()(mod_before)
    # mod_before = AutoDiff(record.requires_grads)(mod_before)
    # mod_before = InferType()(mod_before)
    # mod_before = GradientInputSelection()(mod_before)
    #func_after = InferType()(mod)["main"]
    #print(func_after.astext())

if __name__ == "__main__":
    # pytest.main([__file__])
    test_shardOpAttrs()
    # shardspec = ShardSpec(False, get_dist_device_array(), [8, 1], [2, 1])
    # print(shardspec)