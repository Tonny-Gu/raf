# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
import raf
import numpy as np
from raf.model.model import trace_memory, get_peak_memory
from raf.testing import get_transformer_model, randint, randn_torch
from raf._core.core_utils import str2dev
from raf._core.executor import interpret
from raf._op.imp import matmul
from raf.distributed.sharding import (
    ShardSpecValue,
    ReplicatedSpecValue,
    TupleSpecValue,
    BaseSpecValue,
    ShardOpCallAttrs,
)
from raf._ffi.pass_ import SetShardOpCallAttrs, ToGraphNormalForm, ExpandShardOpCall, InferType
from raf._lib import relay
from raf.testing import randn
from raf.hybrid.hybrid import _make_argument, _unwrap
from raf.testing.common import get_dist_info
from tvm.relay.analysis.analysis import post_order_visit
from tvm.runtime.ndarray import device

def test_shard_add():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            return z

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    attrs = ShardOpCallAttrs(
        TupleSpecValue([ReplicatedSpecValue(), ReplicatedSpecValue()]), ShardSpecValue([3, 2, 1, 0], [2, 2], [1, 2])
    )

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    attrs_map = {call_list[0]: attrs}

    mod0 = SetShardOpCallAttrs(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = ExpandShardOpCall()(mod1)
    print(raf._ffi.ir.AsText(mod2))
    call = relay.Call(op=mod2["main"], args=[_make_argument(x) for x in (m_x, m_y)])
    result = _unwrap(interpret(call, mod2))
    print(result)

def test_reshard_r2s():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            return z
    n_x = np.arange(16).reshape((4, 4))
    m_x = raf.array(n_x)
    m_y = raf._reshard_r2s(m_x, ShardSpecValue([0, 1, 2, 3], [2, 2], [1, 2]))
    print(m_x)
    print(m_y)
    
def test_shard_matmul():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            # s_x = raf._reshard_r2s(x, ShardSpecValue([0, 1, 2, 3], [1, 4], [1, 1]))
            # s_y = raf._reshard_r2s(y, ShardSpecValue([0, 1, 2, 3], [4, 1], [1, 1]))
            s_z = raf.matmul(s_x, s_y)
            # z = raf.allreduce([s_z], "sum")
            return s_z
    
    dctx = raf.distributed.get_context()
    device = "cuda(%s)" % dctx.local_rank
    model = Model()
    model.to(device=device)
    m_x = raf.array(np.arange(16, dtype="float32").reshape((4, 4)), device=device)
    m_y = raf.array(np.ones(16, dtype="float32").reshape((4, 4)), device=device)
    print(m_x)
    print(m_y)
    record = model._internal(m_x, m_y)

    mod_before = record.mod

    attrs = ShardOpCallAttrs(
        TupleSpecValue([ShardSpecValue([0, 1, 2, 3], [1, 4], [1, 1]), ShardSpecValue([0, 1, 2, 3], [4, 1], [1, 1])]),
        ReplicatedSpecValue()
    )

    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    
    attrs_map = {call_list[2]: attrs}

    mod0 = SetShardOpCallAttrs(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = ExpandShardOpCall()(mod1)
    mod3 = InferType()(mod2)

    print(raf._ffi.ir.AsText(mod3))

    call = relay.Call(op=mod3["main"], args=[_make_argument(x) for x in (m_x, m_y)])
    result = _unwrap(interpret(call, mod3))
    print(result)

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_shard_add()
    # test_reshard_r2s()
    # test_shard_matmul()

    model, _ = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")
    model.train_mode() 

    r_x, _ = randint((32, 128), low=0, high=10000, dtype="int64")
    

    

    # a = np.arange(4, dtype="float").reshape((4, 1))
    # b = np.arange(4, dtype="float").reshape((1, 4))
    # ma = raf.array(a)
    # mb = raf.array(b)
    # c = raf.matmul(ma, mb)
    # print(c)
