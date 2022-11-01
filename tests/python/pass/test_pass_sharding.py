# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access, no-self-use, too-many-locals
import numpy as np
import pytest
import raf
from raf._core.ir_ext import extended_var
from raf.distributed.sharding import ShardOpCallAttrs
from raf._ffi.pass_ import (
    AnnotateShardOpCall,
    ToGraphNormalForm,
    ExpandShardOpCall,
    InferType,
    InferShardSpec,
)
from raf._ffi.op import GetOp
from raf._lib import relay
from raf.distributed.sharding import make_replicated_spec, make_shard_spec, make_unset_spec
from raf.testing.utils import run_infer_type
import tvm
from tvm.ir import structural_equal
from tvm.relay import Call
from tvm.relay.analysis.analysis import post_order_visit


def get_opcalls(mod):
    calls = []
    post_order_visit(
        mod["main"].body,
        lambda op: calls.append(op) if isinstance(op, relay.Call) else None,
    )
    return calls

def test_shardspec():
    a = make_shard_spec([4], ranks=4)
    b = make_shard_spec([4], ranks=4)
    assert structural_equal(a, b)

    c = make_shard_spec([2, 2], ranks=4)
    assert not structural_equal(a, c)

    d = make_shard_spec([4], ranks=8)
    assert not structural_equal(a, d)

    e = make_unset_spec()
    f = make_unset_spec()
    assert structural_equal(e, f)
    assert not structural_equal(a, e)

    g = make_shard_spec([4], [4], ranks=4)
    h = make_replicated_spec(ndim=1, ranks=4)
    assert not structural_equal(a, g)
    assert structural_equal(g, h)

    i = make_shard_spec([4], ranks=4, mutable=False)
    assert not structural_equal(a, i)


def test_infer_hint_without_prev_spec():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            b = raf.relu(a)
            return b

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    calls = get_opcalls(mod_before)

    attrs_map = {
        calls[1]: ShardOpCallAttrs(
            [make_unset_spec()], [make_shard_spec([4, 1], ranks=4, mutable=False)]
        ),
        calls[2]: ShardOpCallAttrs(
            [make_unset_spec()], [make_replicated_spec(2, mutable=False)]
        ),
    }

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    mod3 = InferShardSpec()(mod2)
    mod4 = InferType()(mod3)
    mod5 = ExpandShardOpCall()(mod4)
    print("after expand shard opcall")
    print(raf._ffi.ir.AsText(mod5))


def test_infer_hint_inserting_reshard():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            b = raf.relu(a)
            return b

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    calls = get_opcalls(mod_before)
    spec = make_shard_spec([2, 2], [1, 2], 4, mutable=False)

    attrs_map = {
        calls[0]: ShardOpCallAttrs([make_unset_spec(), make_unset_spec()], [make_unset_spec()]),
        calls[1]: ShardOpCallAttrs([make_unset_spec()], [spec]),
    }

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    mod3 = InferShardSpec()(mod2)
    mod4 = InferType()(mod3)
    print("after infer type")
    print(raf._ffi.ir.AsText(mod4))
    mod5 = ExpandShardOpCall()(mod4)
    print("after expand shard opcall")
    print(raf._ffi.ir.AsText(mod5))
    mod6 = InferType()(mod5)
    print("after infer type2")
    print(raf._ffi.ir.AsText(mod6))

    def expected():
        """
        def @main(%x: Tensor[(4, 4), float64], %y: Tensor[(4, 4), float64]) -> Tensor[(2, 4), float64] {
            %0 = raf.op.add(%x, %y) /* ty=Tensor[(4, 4), float64] */;
            %1 = raf.op.strided_slice(%0, [0, 0], [2, 4], [1, 1], str"end") /* ty=Tensor[(2, 4), float64] */;
            %2 = raf.op.relu(%1) /* ty=Tensor[(2, 4), float64] */;
            raf.op.relu(%2) /* ty=Tensor[(2, 4), float64] */
            }
        """
        # x = raf.ir.var("x", shape=(4, 4), dtype="float64")
        # y = raf.ir.var("y", shape=(4, 4), dtype="float64")
        x = extended_var("x", shape=(4, 4), dtype="float64")
        y = extended_var("y", shape=(4, 4), dtype="float64")
        # v0 = raf.ir.op.add(x, y)
        v0 = Call(GetOp("raf.op.add"), [x, y])
        v1 = raf.ir.op.strided_slice(v0, raf.ir.const([0, 0]), raf.ir.const([2, 4]), raf.ir.const([1, 1]), raf.ir.const("end"))
        v2 = raf.ir.op.relu(v1)
        v3 = raf.ir.op.relu(v2)
        return tvm.IRModule.from_expr(relay.Function([x, y], v3))
    
    func_expected = run_infer_type(expected())
    print("expected")
    print(raf._ffi.ir.AsText(func_expected))
    print(tvm.ir.structural_equal(mod6["main"].body, func_expected["main"].body))
    calls1 = get_opcalls(mod6)
    calls2 = get_opcalls(func_expected)
    for i in range(len(calls1)):
        # print()
        # print(raf._ffi.ir.AsText(calls1[i]))
        # print()
        # print(raf._ffi.ir.AsText(calls2[i]))
        print()
        print(repr(calls1[i]))
        print()
        print(repr(calls2[i]))
        print(tvm.ir.structural_equal(calls1[i], calls2[i]))


if __name__ == "__main__":
    test_infer_hint_inserting_reshard()
    # pytest.main([__file__])
