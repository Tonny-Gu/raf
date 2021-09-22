import pytest
import numpy as np
import mnm
from mnm._core.executor import VMExecutor
from mnm.testing import check, compile_vm_model, run_vm_model, get_arr_addr, get_device_list, randn


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_vm(device, shape):
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

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    mod = model._internal(m_x).mod
    executor = VMExecutor(mod, device)
    m_z = executor.make_executor()(m_x).numpy()
    ref_z = model(m_x).numpy()
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'

    # execute 2nd time to reuse the op env
    m_z = executor.vm.run(m_x).numpy()
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_cuda_graph(shape):
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

    dev = "cuda"
    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=dev)
    mod = model._internal(m_x).mod
    executor = VMExecutor(mod, dev, enable_cuda_graph=True)
    m_z = executor.make_executor()(m_x)
    ref_z = model(m_x).numpy()
    np.testing.assert_allclose(m_z.numpy(), ref_z, rtol=1e-5, atol=1e-5)

    m_x2, _ = randn(shape, device=dev)
    m_z2 = executor.vm.run(m_x2)
    ref_z2 = model(m_x2).numpy()
    np.testing.assert_allclose(m_z2.numpy(), ref_z2, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_tuple(device, shape):
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(x, x)
            z = mnm.add(x, y)
            return y, z

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    mod = model._internal(m_x).mod
    executor = VMExecutor(mod, device)
    m_y, m_z = executor.make_executor()(m_x)
    m_y, m_z = m_y.numpy(), m_z.numpy()
    ref_y, ref_z = model(m_x)
    ref_y, ref_z = ref_y.numpy(), ref_z.numpy()
    np.testing.assert_allclose(m_y, ref_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_memory(device, shape):
    # pylint: disable=protected-access
    dtype = 'float32'
    x = mnm.array(np.random.randn(*shape).astype(dtype), device=device)
    t_1 = mnm.array(np.ones(shape, dtype=dtype) * 3)
    t_2 = mnm.array(np.ones(shape, dtype=dtype) * 4)
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.relu(x)
            return y

    model = Model()
    args = [x]
    mod = model._internal(*args).mod
    executor = VMExecutor(mod, device)
    y = executor.make_executor()(*args)
    out = mnm.add(t_1, t_2)
    assert get_arr_addr(out) != get_arr_addr(y)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_simple_fusion(device, shape):
    # pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
    def check_e2e(model, device, args):
        out_before = run_vm_model(model, device, args, disable_fusion=True)
        out_after = run_vm_model(model, device, args)
        check(out_before, out_after)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, x)
            z = mnm.relu(y)
            return z

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    check_e2e(model, device, [m_x])


@pytest.mark.parametrize("device", get_device_list())
def test_split_fusion(device):
    # pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
    shape = [3, 3]

    def check_e2e(model, device, args):
        out_before = run_vm_model(model, device, args)
        out_after = run_vm_model(model, device, args)
        check(out_before, out_after)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.split(x, indices_or_sections=3, axis=0)
            y = y[0]
            z = mnm.relu(y)
            return z

    model = Model()
    m_x, _ = randn(shape, device=device)
    check_e2e(model, device, [m_x])

def test_reshape():
    # pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
    shape = [3, 4, 5]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.reshape(x, (12, 5))
            y = mnm.expand_dims(y, axis=1)
            y = mnm.relu(y)
            y = y + y
            y = mnm.reshape(y, (3, 4, 5))
            return y

    model = Model()
    device = "cpu"
    m_x, _ = randn(shape, device=device)
    with mnm.ir.PassContext(disabled_pass=["FuseTVM", "FuseDialect"]):
        # Disable fusion pass to prevent them from being fused.
        bytecode = compile_vm_model(model, device, [m_x])
    assert bytecode.count("set_shape") == 3

    # Enable memory plan and disable fusion.
    out = run_vm_model(model, device, [m_x], opt_level=3)
    ref_out = model(m_x)
    check(out, ref_out)


if __name__ == "__main__":
    pytest.main([__file__])
