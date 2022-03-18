# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
import raf
import numpy as np
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
from raf._lib import relay
from raf.testing import randn
from raf.hybrid.hybrid import _make_argument, _unwrap
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
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    attrs = ShardOpCallAttrs(
        TupleShardSpec([ReplicatedSpec(), ReplicatedSpec()]), ShardSpec([3, 2, 1, 0], [2, 2], [1, 2])
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
    m_y = raf._reshard_r2s(m_x, ShardSpec([0, 1, 2, 3], [2, 2], [1, 2]))
    print(m_x)
    print(m_y)
    

if __name__ == "__main__":
    # pytest.main([__file__])
    test_ShardOpCallAttrs()
    # test_reshard_r2s()
