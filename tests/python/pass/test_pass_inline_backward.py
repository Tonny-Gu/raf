# pylint: disable=protected-access,invalid-name,attribute-defined-outside-init,no-self-use
import pytest
import tvm
from tvm import relay
import mnm
from mnm.testing import randn


def test_basic():
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    def expected(shape):
        # pylint: disable=too-many-locals
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        dy = relay.var("dy")
        a1 = relay.var("a1")
        x1 = relay.var("x1")
        x2 = relay.var("x2")
        x3 = relay.var("x3")
        x4 = relay.var("x4")
        x5 = relay.var("x5")
        x6 = relay.var("x6")
        gradient = relay.var("gradient")
        ret = relay.var("ret")

        let9 = relay.Let(ret, relay.Tuple([a1, gradient]), ret)
        let8 = relay.Let(gradient, relay.Tuple([x3, x6]), let9)
        let7 = relay.Let(x6, mnm.ir.op.sum(dy, x4, x5), let8)
        let6 = relay.Let(x5, mnm.ir.op.get_kept_dims(dy, y), let7)
        let5 = relay.Let(x4, mnm.ir.op.get_reduce_axis(dy, y), let6)
        let4 = relay.Let(x3, mnm.ir.op.sum(dy, x1, x2), let5)
        let3 = relay.Let(x2, mnm.ir.op.get_kept_dims(dy, x), let4)
        let2 = relay.Let(x1, mnm.ir.op.get_reduce_axis(dy, x), let3)
        let1 = relay.Let(a1, mnm.ir.op.add(x, y), let2)
        return relay.Function([x, y, dy], let1)

    shape = (4, 5)
    model = Add()
    model.train_mode()
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)
    m_x.requires_grad = True
    m_y.requires_grad = True
    record = model._internal(m_x, m_y)
    mod = record.mod
    mod = mnm._ffi.pass_.AutoDiff(record.requires_grads)(mod)
    inlined_func = mnm._ffi.pass_.InlineBackward()(mod)["main"]
    print(inlined_func, expected(shape))
    assert tvm.ir.structural_equal(inlined_func, expected(shape))


def test_no_backward():
    class Model1(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    # model that returns a tuple
    class Model2(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.split(mnm.add(x, y), 2)

    # Get a Relay func
    shape = (4, 5)
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)

    model1 = Model1()
    mod = model1._internal(m_x, m_y).mod
    func = mod["main"]
    inlined_func = mnm._ffi.pass_.InlineBackward()(mod)["main"]
    assert tvm.ir.structural_equal(inlined_func, func)

    model2 = Model2()
    mod = model2._internal(m_x, m_y).mod
    func = mod["main"]
    inlined_func = mnm._ffi.pass_.InlineBackward()(mod)["main"]
    assert tvm.ir.structural_equal(inlined_func, func)


if __name__ == "__main__":
    pytest.main([__file__])
