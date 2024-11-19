#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/11/18 下午3:45 
* Project: AI_Intro 
* File: util.py
* IDE: PyCharm 
* Function:
"""
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


# 日志记录函数
def log_progress(epoch, avg_loss, accuracy):
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"Epoch={epoch + 1}, Loss={avg_loss:.4f}, Val_Acc={accuracy:.4f}\n")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, exp, patience=5, delta=1e-4):
    train_losses = []
    val_accuracies = []
    progress_bar = tqdm(range(num_epochs), desc="Epoch Progress", ncols=150)

    loss_queue = deque(maxlen=patience)  # 用于存储最近几次的损失值 - 早停法

    for epoch in progress_bar:
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        if scheduler is not None:
            scheduler.step()

        predictions, ground_truths = test(model, test_loader, device)
        accuracy = np.sum(np.array(predictions) == np.array(ground_truths)) / len(ground_truths)
        val_accuracies.append(accuracy)

        progress_bar.set_postfix({
            "Epoch": epoch + 1,
            "Loss": f"{avg_loss:.4f}",
            "Validation Acc": f"{accuracy:.4f}"
        })

        # 记录日志信息
        log_progress(epoch, avg_loss, accuracy)

        # 早停检查
        loss_queue.append(avg_loss)
        if len(loss_queue) == patience:
            # 检查最近几次损失的变化是否小于阈值
            if max(loss_queue) - min(loss_queue) < delta:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    # 绘制训练曲线
    plot_training_curves(train_losses, val_accuracies, exp)

    # 保存模型权重
    model_name = model.__class__.__name__  # 获取模型类名
    weight_path = f"{model_name}_weights.pth"
    torch.save(model.state_dict(), weight_path)

def plot_training_curves(train_losses, val_accuracies, exp):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'fig/{exp}_training.png')
    plt.show()


def test(model, test_loader, device):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    return predictions, ground_truths
