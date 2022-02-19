# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""resnet model for CIFAR-10"""
# pylint: disable=attribute-defined-outside-init, protected-access
import torch
import torch.nn as nn

import mnm
from mnm.model import Conv2d, Linear, Sequential
from . import resnet
from .common import randn_torch, one_hot_torch


def param_map(m_model, t_model):
    """maps from m_model parameter name to t_model parameter value"""
    assert m_model.num_blocks == t_model.num_blocks
    res = {
        "conv1.w": t_model.conv1.weight,
        "linear.w": t_model.linear.weight,
        "linear.b": t_model.linear.bias,
    }
    res.update(resnet._param_map(m_model, t_model))
    return res


def init(m_model, t_model, device="cpu"):
    """initialize meta model with parameters of torch model"""
    resnet._init(m_model, t_model, device, param_map)


def check_params(m_model, t_model, atol=1e-3, rtol=1e-3):
    """check the parameters of m_model and t_model"""
    resnet._check_params(m_model, t_model, atol, rtol, param_map)


class MNMResNet50(mnm.Model):
    """meta ResNet50"""

    # pylint: disable=missing-function-docstring, too-many-instance-attributes
    def build(self, num_blocks, num_classes=10):
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = Linear(512 * resnet.MNMBottleneck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for one_stride in strides:
            layers.append(resnet.MNMBottleneck(self.in_planes, planes, one_stride))
            self.in_planes = planes * resnet.MNMBottleneck.expansion
        return Sequential(*layers)

    @mnm.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = mnm.log_softmax(y_pred)
        loss = mnm.nll_loss(y_true, y_pred)
        return loss

    @mnm.model.trace
    def forward_infer(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = mnm.avg_pool2d(out, 4, 4)
        out = mnm.batch_flatten(out)
        out = self.linear(out)
        return out


class TorchResNet50(nn.Module):
    """torch ResNet50"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method, missing-function-docstring
    def __init__(self, num_blocks, num_classes=10):
        super(TorchResNet50, self).__init__()
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * resnet.TorchBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for one_stride in strides:
            layers.append(resnet.TorchBottleneck(self.inplanes, planes, one_stride))
            self.inplanes = planes * resnet.TorchBottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y_true):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4, 4)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.linear(x)
        y_pred = x
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


def get_model(layers):
    """get resnet model"""
    m_model = MNMResNet50(layers)
    t_model = TorchResNet50(layers)
    init(m_model, t_model)
    m_model.train_mode()
    t_model.train()
    return m_model, t_model


def get_input(batch_size=1, device="cuda"):
    """get resnet input"""
    m_x, t_x = randn_torch([batch_size, 3, 32, 32], device=device, requires_grad=True)
    m_y, t_y = one_hot_torch(batch_size, num_classes=10, device=device)
    return [(m_x, m_y), (t_x, t_y)]
