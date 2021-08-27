# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import mnm
from mnm.testing import run_infer_type, randn
import tvm
from tvm import relay


def test_simple():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.c)
            y = mnm.relu(y)
            y = mnm.log(y)
            return y

    def expected():
        x = relay.var("x", shape=(10, 20))
        c = relay.var("c", shape=(1,))
        y = mnm.ir.op.add(x, c)
        y = mnm.ir.op.log(mnm.ir.op.relu(y))
        f = relay.Function([x, c], y)
        return f

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_tuple():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            zz = mnm.split(z, 2)
            return zz[0]

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        z = mnm.ir.op.add(x, y)
        z = mnm.ir.op.split(z, 2)
        z = relay.TupleGetItem(z, 0)
        f = relay.Function([x, y], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_diamond():
    konst, _ = randn((1,))
    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x, y):
            z1 = mnm.add(x, y)
            z2 = mnm.multiply(x, self.c)
            return mnm.relu(mnm.add(z1, z2))

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        c = relay.var("c", shape=(1,))
        z1 = mnm.ir.op.add(x, y)
        z2 = mnm.ir.op.multiply(x, c)
        z = mnm.ir.op.add(z1, z2)
        z = mnm.ir.op.relu(z)
        f = relay.Function([x, y, c], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
