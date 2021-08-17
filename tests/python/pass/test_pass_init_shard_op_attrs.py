# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import mnm
from mnm._ffi.pass_ import AutoDiff, GradientInputSelection, InferType, InitShardOpAttrs
from mnm._lib import tvm
from mnm._lib import relay
from mnm.testing import randn, run_infer_type
from mnm._core.module import IRModule

def test_shardOpAttrs():

    class Model(mnm.Model):
        def build(self):
            self.w, _ = randn((1, 1, 3, 3))
            self.w.requires_grad = True

        @mnm.model.trace
        def forward(self, x):
            y = mnm.conv2d(x, self.w)
            z = mnm.relu(y)
            return z

    model = Model()
    m_x, _ = randn((1, 1, 224, 224))
    m_x.requires_grad = True
    record = model._internal(m_x)
    mod_before = record.mod
    # mod_before = InferType()(mod_before)
    # mod_before = AutoDiff(record.requires_grads)(mod_before)
    # mod_before = InferType()(mod_before)
    # mod_before = GradientInputSelection()(mod_before)
    mod = InitShardOpAttrs()(mod_before)
    func_after = InferType()(mod)["main"]
    print(func_after.astext())

if __name__ == "__main__":
    # pytest.main([__file__])
    test_shardOpAttrs()
