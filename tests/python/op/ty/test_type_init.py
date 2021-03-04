# pylint: disable=protected-access
import pytest
import mnm
from mnm.testing import randint, check_type, run_infer_type
from tvm.relay import TensorType, FuncType


@pytest.mark.parametrize("op", [mnm._op.sym.zeros, mnm._op.sym.ones])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "bool"])
def test_init_ops(op, shape, dtype):
    # pylint: disable=invalid-name, attribute-defined-outside-init
    class InitOpModel(mnm.Model):
        def build(self, op, shape, dtype):
            self.op = op
            self.shape = shape
            self.dtype = dtype

        @mnm.model.trace
        def forward(self):
            return self.op(shape=self.shape, dtype=self.dtype)

    model = InitOpModel(op, shape, dtype)
    m_func = model._internal().func
    m_func = run_infer_type(m_func)
    ty = TensorType(shape, dtype=dtype)
    desired_type = FuncType([], ty)
    check_type(m_func, desired_type)


@pytest.mark.parametrize("indices_shape", [(1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
def test_one_hot(indices_shape, depth, dtype):
    # pylint: disable=invalid-name, attribute-defined-outside-init
    class OneHotModel(mnm.Model):
        def build(self, depth, dtype):
            self.depth = depth
            self.dtype = dtype

        @mnm.model.trace
        def forward(self, indices, on_value, off_value):
            return mnm.one_hot(indices, on_value, off_value, depth=self.depth, dtype=self.dtype)

    model = OneHotModel(depth, dtype)
    m_indices, _ = randint(shape=indices_shape, high=10)
    m_on_value = mnm.array(1, dtype="int32")
    m_off_value = mnm.array(0, dtype="int32")
    m_func = model._internal(m_indices, m_on_value, m_off_value).func
    m_func = run_infer_type(m_func)
    indices_ty = TensorType(indices_shape, dtype="int64")
    value_ty = TensorType((), dtype="int32")
    m_shape = list(indices_shape)
    m_shape.append(depth)
    y_ty = TensorType(m_shape, dtype=dtype)
    desired_type = FuncType([indices_ty, value_ty, value_ty], y_ty)
    check_type(m_func, desired_type)


if __name__ == "__main__":
    pytest.main([__file__])