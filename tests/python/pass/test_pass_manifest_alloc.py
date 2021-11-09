import pytest
import mnm
from mnm._lib import tvm
from mnm._core.module import IRModule
from mnm._core.device import Device
from mnm.testing import get_device_list, randn


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_memory_alloc(device, shape):
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(x, x)
            z = mnm.add(x, y)
            return z

    model_before = Model()
    model_before.infer_mode()
    m_x, _ = randn(shape, device=device)
    func = model_before._internal(m_x).mod['main']
    mod = IRModule.from_expr(func)
    mod = mnm._ffi.pass_.InferType()(mod)
    with Device(device if device != 'cpu' else 'llvm'):
        mod = mnm._ffi.pass_.ManifestAlloc()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)
    text = mod['main'].astext()
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "invoke_op" in text


def test_dynamic_model():
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.argwhere(x)
            y = mnm.argwhere(y)
            y = mnm.split(y, 2)
            y = mnm.add(y[0], y[1])
            y = mnm.abs(y)
            return y

    model = Model()
    m_x, _ = randn((2, 2))
    mod = model._internal(m_x).mod
    with tvm.transform.PassContext():
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
        mod = mnm._ffi.pass_.FuseTVM()(mod)
        mod = mnm._ffi.pass_.ToANormalForm()(mod)
        mod = mnm._ffi.pass_.InlinePrimitives()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
        mod = mnm._ffi.pass_.ManifestAlloc()(mod)
    text = mnm.ir.AsText(mod["main"])
    assert '\n'.join(text.splitlines()[:-4]) == """#[version = "0.0.5"]
fn (%x: Tensor[(2, 2), float32]) -> Tensor[(meta[tir.Div][0], 2), int32] {
  let %x_0 = mnm.op.vm.alloc_storage(int64(32), int64(64), int32(1), int32(0), str"int32");
  let %x_1 = mnm.op.vm.alloc_tensor(%x_0, [4, 2], str"int32", [4, 2]);
  let %x_2 = mnm.op.vm.alloc_storage(int64(16), int64(64), int32(1), int32(0), str"int64");
  let %x_3 = mnm.op.vm.alloc_tensor(%x_2, [2], str"int64", [2]);
  let %x_4 = mnm.op.upper_bound.argwhere;
  let %x_5 = (%x,);
  let %x_6 = (%x_1, %x_3);
  let %x_7 = mnm.op.vm.invoke_op(%x_4, %x_5, %x_6);
  let %x1 = mnm.op.vm.set_shape(%x_1, %x_3);
  let %x_8 = (%x1,);
  let %x_9 = mnm.op.upper_bound.argwhere;
  let %x_10 = mnm.op.vm.infer_type(%x_9, %x_8);
  let %x_11 = %x_10.1;
  let %x_12 = %x_11.0;
  let %x_13 = %x_11.1;
  let %x_14 = mnm.op.vm.alloc_storage(%x_13, int64(64), int32(1), int32(0), str"int32");
  let %x_15 = mnm.op.vm.alloc_tensor(%x_14, %x_12, str"int32", %x_12);
  let %x_16 = %x_10.2;
  let %x_17 = %x_16.0;
  let %x_18 = %x_16.1;
  let %x_19 = mnm.op.vm.alloc_storage(%x_18, int64(64), int32(1), int32(0), str"int64");
  let %x_20 = mnm.op.vm.alloc_tensor(%x_19, %x_17, str"int64", %x_17);
  let %x_21 = (%x_15, %x_20);
  let %x_22 = mnm.op.vm.invoke_op(%x_9, %x_8, %x_21);
  let %x2 = mnm.op.vm.set_shape(%x_15, %x_20);
  let %x_23 = nullptr;
  let %x_24 = nullptr;
  let %x_25 = (%x2, %x_23, %x_24);
  let %x_26 = fn (%p0: Tensor[(?, 2), int32], %p1: (), %p2: (), Primitive=1, Dialect="tvm") -> Tensor[(meta[tir.Div][0], 2), int32] {
    %0 = mnm.op.tvm.split(%p0, int64(2), int64(0)) /* ty=(Tensor[(meta[tir.Div][0], 2), int32], Tensor[(meta[tir.Div][1], 2), int32]) */;
    %1 = %0.0;
    %2 = %0.1;
    %3 = mnm.op.tvm.add(%1, %2, %p1, %p2) /* ty=Tensor[(meta[tir.Div][0], 2), int32] */;
    mnm.op.tvm.abs(%3) /* ty=Tensor[(meta[tir.Div][0], 2), int32] */
  };
  let %x_27 = mnm.op.vm.infer_type(%x_26, %x_25);
  let %x_28 = %x_27.1;
  let %x_29 = %x_28.0;
  let %x_30 = %x_28.1;
  let %x_31 = mnm.op.vm.alloc_storage(%x_30, int64(64), int32(1), int32(0), str"int32");
  let %x_32 = mnm.op.vm.alloc_tensor(%x_31, %x_29, str"int32", %x_29);
  let %x_33 = %x_27.0;
  let %x_34 = (%x_32,);
  let %x_35 = mnm.op.vm.invoke_op(%x_33, %x_25, %x_34);
  let %x4 = %x_32;
  %x4
}"""

def test_reshape():
    # pylint: disable=protected-access, no-self-use
    shape = [3, 4, 5]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.reshape(x, (12, 5))
            y = mnm.expand_dims(y, axis=0)
            y = mnm.squeeze(y)
            y = mnm.relu(y)
            y = mnm.reshape(y, (3, 4, 5))
            return y

    model = Model()
    m_x, _ = randn(shape, device="cpu")
    func = model._internal(m_x).mod['main']
    mod = IRModule.from_expr(func)
    mod = mnm._ffi.pass_.InferType()(mod)
    with Device("cpu"):
        mod = mnm._ffi.pass_.ManifestAlloc()(mod)
    text = mod['main'].astext()
    assert text.count("vm.set_shape") == 4
    assert "reshape" not in text
    assert "expand_dims" not in text
    assert "squeeze" not in text


if __name__ == "__main__":
    pytest.main([__file__])
