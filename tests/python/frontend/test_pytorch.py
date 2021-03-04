# pylint:disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint:disable=not-callable,abstract-method,too-many-locals
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mnm
from mnm._op import sym
from mnm.frontend import from_pytorch
from mnm.testing import randn_torch, check, one_hot_torch, run_vm_model


class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               padding=2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16,
                                 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape_dict", [{"input0": ((32, 3, 28, 28), "float32")}])
@pytest.mark.parametrize("mode", ["forward", "backward", "sgd"])
def test_lenet(shape_dict, mode):
    device = "cuda"
    input_shape = list(shape_dict.values())[0][0]
    batch_size = input_shape[0]

    # Prepare two models.
    t_model = TorchLeNet(input_shape[2])
    m_model = from_pytorch(t_model, shape_dict)

    # Set the target device.
    t_model.to(device=device)
    m_model.to(device=device)

    # Prepare data.
    m_x, t_x = randn_torch(input_shape, device=device)

    if mode == "forward":
        m_model.infer_mode()
        t_model.eval()
        m_y = m_model(m_x)
        t_y = t_model(t_x)
        check(m_y, t_y, rtol=1e-4, atol=1e-4)
        return

    m_ytrue, t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=10, device=device)
    m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False)

    # append loss function
    out = m_model.record(m_x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    m_model = m_model + loss

    if mode == "backward":
        m_x.requires_grad = True

        m_model.train_mode()
        t_model.train()

        m_loss = m_model(m_x, m_ytrue)

        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)

        check(m_loss, t_loss)

        m_loss.backward()
        t_loss.backward()
        check(m_loss, t_loss, rtol=1e-4, atol=1e-4)
    else:
        assert mode == "sgd"

        m_model.train_mode()
        m_model.to(device=device)

        m_trainer = mnm.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
        m_loss = run_vm_model(m_trainer, device, [m_dy, m_x, m_ytrue])[0]

        t_trainer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
        t_model.train()

        t_trainer.zero_grad()
        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)
        t_loss.backward(t_dy)
        t_trainer.step()
        check(m_loss, t_loss)


if __name__ == "__main__":
    pytest.main([__file__])