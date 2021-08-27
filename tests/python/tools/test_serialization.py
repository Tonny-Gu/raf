# pylint: disable=protected-access
import pytest
import tvm
import mnm
from tvm import relay
from mnm.testing import check, randn
from mnm._core.ir_ext import extended_var
from mnm._core.module import IRModule
from mnm._ffi.model import RunModel
from mnm.model.trace import _unwrap


def test_ext_constant():
    def expected():
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = mnm._core.value.TensorValue.from_numpy(x_const.numpy())
        x = mnm._ffi.ir._make.Constant(x_value)
        pooled = mnm.ir.op.max_pool2d(x, 3, 1, 0)
        ovar = extended_var("out")
        let = relay.Let(ovar, pooled, ovar)
        return relay.Function([], let)

    func_origin = expected()
    mod_origin = IRModule.from_expr(func_origin)
    out_origin = _unwrap(RunModel(mod_origin, []))

    json = mnm.ir.save_json(func_origin)

    func_loaded = tvm.ir.load_json(json)
    mod_loaded = IRModule.from_expr(func_loaded)
    out_loaded = _unwrap(RunModel(mod_loaded, []))

    assert tvm.ir.structural_equal(func_loaded, func_origin)
    check(out_origin, out_loaded)


if __name__ == "__main__":
    pytest.main([__file__])
