#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/19 下午2:54 
* Project: AI_Intro 
* File: resnet.py
* IDE: PyCharm 
* Function: ResNet model
"""
import torch
import torch.nn as nn

class ResNet(nn.Module):
    """
    ResNet 通用模型类，包含 ResNet-18、34、50、101 和 152 的实例化方法。
    """
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet 各层
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        创建一个包含多个 block 的 ResNet 层。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # ResNet 模型实例化方法
    @staticmethod
    def resnet18(num_classes=100):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    @staticmethod
    def resnet34(num_classes=100):
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

    @staticmethod
    def resnet50(num_classes=100):
        return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)

    @staticmethod
    def resnet101(num_classes=100):
        return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)

    @staticmethod
    def resnet152(num_classes=100):
        return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)


class BasicBlock(nn.Module):
    """
    ResNet-18 和 ResNet-34 的基本 Block。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        # Shortcut 层，用于匹配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    ResNet-50、101 和 152 的 BottleNeck Block。
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        # Shortcut 层，用于匹配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
