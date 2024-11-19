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
from torch import nn, optim
from torch.utils.data import DataLoader
from util import train
from models import DenseNet, ResNet, VGG

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = {
    "data_root": "./data",
    "batch_size": 128,
    "num_workers": 2,
    "model_name": "ResNet",  # 模型名称，可选 DenseNet, ResNet, VGG
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 100,
    "patience": 5,  # 早停法耐心值
    "delta": 1e-4,  # 早停法最小增量
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "mean": [0.5071, 0.4865, 0.4409],  # CIFAR-100 数据集均值
    "std": [0.2673, 0.2564, 0.2762],  # CIFAR-100 数据集标准差
}


def get_dataset(root, train, transform):
    """获取 CIFAR-100 数据集"""
    download_flag = not os.path.exists(os.path.join(root, 'cifar-100-python'))
    return torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=download_flag,
        transform=transform
    )


if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"])
    ])

    # 加载数据集
    train_set = get_dataset(config["data_root"], train=True, transform=transform)
    test_set = get_dataset(config["data_root"], train=False, transform=transform)

    # 数据加载器
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]
    )

    # 模型选择
    if config["model_name"] == "DenseNet":
        model = DenseNet(growth_rate=32, block_layers=(6, 12, 24, 16))      # DenseNet121
    elif config["model_name"] == "ResNet":
        model = ResNet.resnet101()
    elif config["model_name"] == "VGG":
        model = VGG.vgg19()
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")

    # 训练模型
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        ),
        scheduler=None,
        device=config["device"],
        num_epochs=config["num_epochs"],
        exp="exp2",
        patience=config["patience"],
        delta=config["delta"]
    )
