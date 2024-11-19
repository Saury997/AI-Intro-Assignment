#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/19 下午2:59 
* Project: AI_Intro 
* File: vgg.py
* IDE: PyCharm 
* Function: VGG model
"""
import torch.nn as nn


# 配置字典，表示各版本VGG的结构
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    """VGG 模型定义"""
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        # 特征提取部分
        self.features = features
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        # 是否进行权重初始化
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    @staticmethod
    def vgg11(num_classes=100):
        return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)

    @staticmethod
    def vgg13(num_classes=100):
        return VGG(make_layers(cfg['B'], batch_norm=True), num_classes)

    @staticmethod
    def vgg16(num_classes=100):
        return VGG(make_layers(cfg['D'], batch_norm=True), num_classes)

    @staticmethod
    def vgg19(num_classes=100):
        return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)


def make_layers(cfg, batch_norm=False):
    """根据配置构建网络层"""
    layers = []
    in_channels = 3  # 输入通道数（RGB图像）
    for v in cfg:
        if v == 'M':  # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 卷积层 + 批归一化（可选） + 激活函数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
