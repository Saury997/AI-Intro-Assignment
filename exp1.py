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
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from util import train
from models import MLP, CNN


config = {
    "data_path": "data/mnist.npz",  # MNIST数据集路径
    "batch_size": 128,             # 批量大小
    "learning_rate": 0.001,        # 学习率
    "weight_decay": 1e-5,          # 权重衰减系数
    "step_size": 15,               # 学习率调度步长
    "gamma": 0.1,                  # 学习率衰减因子
    "num_epochs": 50,              # 训练轮次
    "num_classes": 10,             # 分类数
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 计算设备
    "model_name": "CNN",           # 模型名称
    "predict_image_path": "data/exp1_test.jpg",  # 预测图片路径
}

def load_mnist_data(path):
    """加载MNIST数据集"""
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def predict(model, image_path):
    """对单张图像进行预测"""
    model.eval()
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_tensor = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).view(1, -1)  # 调整为 (1, 784)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if __name__ == '__main__':
    # 加载MNIST数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data(config["data_path"])

    # 归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 转换为PyTorch张量
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 初始化模型、损失函数、优化器和学习率调度器
    if config["model_name"] == 'MLP':
        model = MLP(in_channels=28 * 28, num_classes=config["num_classes"])
    elif config["model_name"] == 'CNN':
        model = CNN(in_channels=1, num_classes=config["num_classes"])
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # 模型训练
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["device"],
        num_epochs=config["num_epochs"],
        exp='exp1',
    )

    # 加载训练后的模型权重
    weight_path = f"weights/{config['model_name']}_weights.pth"
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    # 预测单张图片
    res = predict(model.to('cpu'), config["predict_image_path"])
    print("预测测试图像代表的数字为:", res)
