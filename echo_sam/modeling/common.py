# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d, deform_conv2d
import torch.nn.functional as F

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class Adapter(nn.Module):
    def __init__(self, D_features, down_ratio=0.25, act_layer=nn.GELU, skip_connect=True): #0.25
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * down_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        offset = self.offset(x)
        mask = self.mask_conv(x)
        mask = torch.sigmoid(mask)
        x = deform_conv2d(x, offset, self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, mask=mask)
        return x
    
class SideNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(SideNetwork, self).__init__()
        self.deform_conv1 = DeformableConv2d(in_channels, 32, kernel_size, stride)
        self.ln1 = LayerNorm2d(32)
        self.deform_conv2 = DeformableConv2d(32, 64, kernel_size, stride)
        self.ln2 = LayerNorm2d(64)
        self.deform_conv3 = DeformableConv2d(64, 128, kernel_size, stride)
        self.ln3 = LayerNorm2d(128)
        self.deform_conv4 = DeformableConv2d(128, out_channels, kernel_size, stride)
        self.ln4 = LayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.deform_conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.deform_conv2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.deform_conv3(x)
        x = self.ln3(x)
        x = self.act(x)
        x = self.deform_conv4(x)
        x = self.ln4(x)
        x = self.act(x)
        return x

class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2
        
        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        sobel_weight = self.sobel_weight * self.sobel_factor
        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out
    
class Sobel_Enhance_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):
        super(Sobel_Enhance_Layer, self).__init__()
        self.sobel = SobelConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, requires_grad)
        self.relu = nn.ReLU(inplace=True)
        self.concat = nn.Conv2d(in_channels+out_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.sobel(x)
        out = self.relu(out)
        out = torch.cat((x, out), 1)
        out = self.concat(out)
        out = self.relu(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = Sobel_Enhance_Layer(3, 4)
    print(model(x).shape)