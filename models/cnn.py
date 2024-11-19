#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/19 下午3:38 
* Project: AI_Intro 
* File: cnn.py
* IDE: PyCharm 
* Function:
"""
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 第一层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二层卷积
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, num_classes)  # 输出层 (10 类)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x