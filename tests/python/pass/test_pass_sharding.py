# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
import raf
import numpy as np
from raf.distributed.sharding.inferhint import infer_shardspec
from raf.distributed.sharding.utils import make_unset_spec
from raf.model.model import trace_memory, get_peak_memory
from raf.testing import get_transformer_model, randint, randn_torch
from raf._core.core_utils import str2dev
from raf._core.executor import interpret
from raf._op.imp import matmul
from raf.distributed.sharding import (
    ShardSpec,
    BaseShardSpec,
    ShardOpCallAttrs,
)
from raf.distributed.sharding import (
    make_replicated_spec,
    make_shard_spec
)
from raf._ffi.pass_ import AnnotateShardOpCall, ToGraphNormalForm, ExpandShardOpCall, InferType, InferShardSpec
from raf._lib import relay
from raf.testing import randn
from raf.hybrid.hybrid import _make_argument, _unwrap
from tvm.ir import structural_equal
from tvm.relay.analysis.analysis import post_order_visit
from tvm.runtime.ndarray import device

def test_shardspec():
    a = make_shard_spec([1], ranks = 4)
    b = make_shard_spec([1], ranks = 4)
    print(structural_equal(a, b))
    c = make_shard_spec([3], ranks = 4)
    print(structural_equal(a, c))
    d = make_shard_spec([1], ranks = 3)
    print(structural_equal(a, d))
    e = make_unset_spec()
    f = make_unset_spec()
    print(structural_equal(a, e))
    print(structural_equal(e, f))

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

    s_spec = make_shard_spec([2, 2], [1, 2], 4)

    attrs = ShardOpCallAttrs([s_spec, s_spec], [s_spec])

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    attrs_map = {call_list[0]: attrs}

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    print(raf._ffi.ir.AsText(mod0))
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = ExpandShardOpCall()(mod1)
    print(raf._ffi.ir.AsText(mod2))
    call = relay.Call(op=mod2["main"], args=[_make_argument(x) for x in (m_x, m_y)])
    result = _unwrap(interpret(call, mod2))
    print(result)

def test_infer_hint():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            return a

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    s_spec = make_shard_spec([2, 2], [1, 2], 4)

    attrs = ShardOpCallAttrs([s_spec, s_spec], [make_unset_spec()])
    attrs1 = ShardOpCallAttrs([make_unset_spec()], [make_unset_spec()])

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    attrs_map = {call_list[0]: attrs, call_list[1]: attrs1}

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    print(raf._ffi.ir.AsText(mod2))

    mod3 = InferShardSpec()(mod2)
    print(raf._ffi.ir.AsText(mod3))

def test_infer_hint_with_reshard():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            return a

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )

    spec = make_shard_spec([2, 2], [1, 2], 4, mutable=False)

    attrs_map = {
        call_list[0]: ShardOpCallAttrs([make_unset_spec(), make_unset_spec()], [make_unset_spec()]),
        call_list[1]: ShardOpCallAttrs([make_unset_spec()], [spec])
    }

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    print(raf._ffi.ir.AsText(mod2))

    mod3 = InferShardSpec()(mod2)
    print(raf._ffi.ir.AsText(mod3))

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
    m_y = raf._reshard_r2s(m_x, ShardSpec([0, 1, 2, 3], [2, 2], [1, 2]))
    print(m_x)
    print(m_y)
    
def test_shard_matmul():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            s_z = raf.matmul(x, y)
            # z = raf.allreduce([s_z], "sum")
            return s_z
    
    comm = raf.distributed.get_communicator()
    device = "cuda(%s)" % comm.local_rank
    model = Model()
    model.to(device=device)
    m_x = raf.array(np.arange(16, dtype="float32").reshape((4, 4)), device=device)
    m_y = raf.array(np.ones(16, dtype="float32").reshape((4, 4)), device=device)
    s_x = raf._reshard_r2s(m_x, ShardSpec([0, 1, 2, 3], [1, 4], [1, 1], True))
    s_y = raf._reshard_r2s(m_y, ShardSpec([0, 1, 2, 3], [4, 1], [1, 1], True))
    record = model._internal(m_x, m_y)

    mod_before = record.mod

    attrs = ShardOpCallAttrs(
        [ShardSpec([0, 1, 2, 3], [1, 4], [1, 1], True), ShardSpec([0, 1, 2, 3], [4, 1], [1, 1], True)],
        [make_replicated_spec(2, 4)]
    )

    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )
    
    attrs_map = {call_list[0]: attrs}

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = ExpandShardOpCall()(mod1)
    mod3 = InferType()(mod2)

    print(raf._ffi.ir.AsText(mod3))

    call = relay.Call(op=mod3["main"], args=[_make_argument(x) for x in (s_x, s_y)])
    result = _unwrap(interpret(call, mod3))
    print(result)

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_shardspec()
    # test_shard_add()
    # test_reshard_r2s()
    # test_shard_matmul()
    test_infer_hint_with_reshard()

    # model, _ = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")
    # model.to(device="cuda(0)")
    # model.train_mode() 

    # r_x, _ = randint((32, 128), low=0, high=10000, dtype="int64")
    # mod = model._internal(r_x).mod
    # print(raf._ffi.ir.AsText(mod))

    

    

    # a = np.arange(4, dtype="float").reshape((4, 1))
    # b = np.arange(4, dtype="float").reshape((1, 4))
    # ma = raf.array(a)
    # mb = raf.array(b)
    # c = raf.matmul(ma, mb)
    # print(c)
