#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/13 下午6:48 
* Project: AI_Intro 
* File: exp1.py
* IDE: PyCharm 
* Function: Hand-written Digits Recognition
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from util import train


def load_mnist_data(path='data/mnist.npz'):
    """Loads the MNIST dataset"""
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


class MLP(nn.Module):
    def __init__(self):
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


def predict(model, image_path):
    model.eval()
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_tensor = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).view(1, 784)  # (1, 784)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# 导入MNIST数据
(x_train, y_train), (x_test, y_test) = load_mnist_data()

# 缩放
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 转换为PyTorch张量
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 调整数据形状 (batch_size, 1, 28, 28)
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

# 训练集DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 测试集DataLoader
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 初始化模型、损失函数、优化器和其他参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
num_epochs = 50

if __name__ == '__main__':
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, exp='exp1')

    # model = MLP()
    weight_path = f"{model.__class__.__name__}_weights.pth"
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    res = predict(model.to('cpu'), 'data/exp1_test.jpg')
    print("预测测试图像代表的数字为:", res)
