#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/19 下午3:09 
* Project: AI_Intro 
* File: mlp.py
* IDE: PyCharm 
* Function: MLP model
"""
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 1024)  # 输入层到第一个隐藏层 (784 -> 1024)
        self.bn1 = nn.BatchNorm1d(1024)  # 批量归一化
        self.fc2 = nn.Linear(1024, 512)  # 第二个隐藏层 (1024 -> 512)
        self.bn2 = nn.BatchNorm1d(512)  # 批量归一化
        self.fc3 = nn.Linear(512, 128)  # 第三个隐藏层 (512 -> 128)
        self.bn3 = nn.BatchNorm1d(128)  # 批量归一化
        self.fc4 = nn.Linear(128, 10)  # 输出层 (128 -> 10)
        self.dropout = nn.Dropout(0.5)  # Dropout层

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平成 (batch_size, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x
