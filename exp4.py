#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/4 下午7:04 
* Project: AI_Intro 
* File: exp4.py
* IDE: PyCharm 
* Function: Application of Graph Convolutional Networks on Cora Dataset 
"""
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from util import plot_training_curves, log_progress
from models import SplineGCN, GCN, GAT, ChebNet

config = {
    "data_root": "./data",
    "model_name": "SplineGCN",  # 模型名称，可选 SplineGCN, GCN, GAT, ChebNet
    "learning_rate": 0.01,
    "weight_decay": 5e-3,
    "num_epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def get_cora_data(path):
    """获取 Cora 数据"""
    dataset = Planetoid(root=path, name='Cora', transform=T.TargetIndegree())
    data = dataset[0]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:data.num_nodes - 1000] = 1
    data.val_mask = None
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.num_nodes - 500:] = 1

    return data, dataset


def train(data, model, optimizer, device):
    model.to(device).train()
    optimizer.zero_grad()

    out = model(data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


def test():
    model.eval()
    log_probs = model(data)
    accs = []

    # 遍历 train_mask 和 test_mask
    for mask_name in ['train_mask', 'test_mask']:
        mask = data[mask_name]
        pred = log_probs[mask].max(dim=1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs


if __name__ == '__main__':
    # 加载数据
    data, dataset = get_cora_data(config['data_root'])

    # 确定模型
    if config['model_name'] == 'SplineGCN':
        model = SplineGCN(dataset.num_features, dataset.num_classes)
    elif config['model_name'] == 'GCN':
        model = GCN(dataset.num_features, dataset.num_classes)
    elif config['model_name'] == 'GAT':
        model = GAT(dataset.num_features, dataset.num_classes)
    elif config['model_name'] == 'ChebNet':
        model = ChebNet(dataset.num_features, dataset.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")

    # 配置训练设置
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"------Training model {config['model_name']}------")
    progress_bar = tqdm(range(1, config['num_epochs'] + 1), desc="Training Epochs", ncols=150)
    train_time = datetime.now().strftime("%Y%m%d%H%M%S")
    losses, accuracies = [], []

    for epoch in progress_bar:
        loss = train(
            data=data.cuda(),
            model=model,
            optimizer=optimizer,
            device=device
        )
        losses.append(loss)

        train_acc, test_acc = test()
        progress_bar.set_postfix({
            "Epoch": epoch,
            "Loss": f"{loss:.4f}",
            "Train Accuracy": f"{train_acc:.4f}",
            "Test Accuracy": f"{test_acc:.4f}"
        })
        accuracies.append(test_acc)

        # 记录日志信息
        log_progress(epoch, loss, config['model_name'], train_time, test_acc)

    # 绘制训练曲线
    plot_training_curves(losses, accuracies, 'exp4', config['model_name'])
