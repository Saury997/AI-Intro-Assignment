#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/18 下午3:41 
* Project: AI_Intro 
* File: exp2.py
* IDE: PyCharm 
* Function: Image Classification based on CNN
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from util import train

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不同，使用1x1卷积来调整输入的维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 四个残差块
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, stride):
        # 每个 layer 包含两个 BasicBlock
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 展平为一维
        out = out.view(out.size(0), -1)

        # 全连接层输出
        out = self.fc(out)
        return out


class VGG13(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG13, self).__init__()

        # 定义 VGG-13 卷积层（包含多个卷积 + ReLU + 池化）
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 通过一个虚拟输入计算卷积层输出的形状
        self._to_linear = 512 * 8 * 8
        x = torch.randn(1, 3, 32, 32)  # 输入一个随机图像
        self._get_linear_in_features(x)

        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def _get_linear_in_features(self, x):
        """通过卷积层的输出计算全连接层的输入维度"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self._to_linear = x.size(1)

    def forward(self, x):
        # 前向传播
        x = self.features(x)  # 经过卷积层
        x = x.view(x.size(0), -1)  # 展平多维输入为一维
        x = self.classifier(x)  # 通过全连接层
        return x


def get_dataset(root, train, transform):
    download_flag = not os.path.exists(os.path.join(root, 'cifar-100-python'))
    return torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=download_flag,
        transform=transform
    )


# 定义数据预处理（如归一化、数据增强等）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

# 加载训练集和测试集
train_set = get_dataset('./data', train=True, transform=transform)
test_set = get_dataset('./data', train=False, transform=transform)

# 创建数据加载器（DataLoader）以便于批量训练
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# 初始化模型、损失函数、优化器和其他参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG13().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = None
num_epochs = 400


if __name__ == '__main__':
    train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=50,
    exp="exp2",
    patience=10,
    delta=1e-4
)

