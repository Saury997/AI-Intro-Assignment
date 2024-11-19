#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/19 下午2:51 
* Project: AI_Intro 
* File: densenet.py
* IDE: PyCharm 
* Function: DenseNet model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)  # Concatenate along channel dimension
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels=3, growth_rate=32, block_layers=(6, 12, 24, 16), num_classes=100):
        super(DenseNet, self).__init__()
        num_init_features = 2 * growth_rate

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        # 构建 DenseBlock 和 Transition 层
        self.features = nn.ModuleList()
        num_features = num_init_features
        for i, num_layers in enumerate(block_layers):
            self.features.append(DenseBlock(num_layers, num_features, growth_rate))
            num_features += num_layers * growth_rate
            if i != len(block_layers) - 1:  # 最后一层不需要 Transition
                self.features.append(Transition(num_features, num_features // 2))
                num_features = num_features // 2

        # 最后批归一化
        self.bn = nn.BatchNorm2d(num_features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.features:
            x = layer(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 自适应池化到 1x1
        x = torch.flatten(x, 1)  # 展平
        x = self.fc(x)
        return x
    