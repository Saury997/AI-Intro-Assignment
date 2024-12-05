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
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
# import cartopy.crs as ccrs
# import geopandas as gpd


def log_progress(epoch, avg_loss, model_name, time, accuracy: float = None):
    with open(f"log/training{model_name}_{time}.txt", "a") as log_file:
        if accuracy is None:
            log_file.write(f"Epoch={epoch + 1}, Loss={avg_loss:.4f}\n")
        else:
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


def train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs, exp, test_loader=None, patience=5,
          delta=1e-4):
    model.to(device)
    train_losses = []
    val_accuracies = []
    progress_bar = tqdm(range(num_epochs), desc="Epoch Progress", ncols=150)
    model_name = model.__class__.__name__  # 获取模型类名
    train_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    loss_queue = deque(maxlen=patience)  # 用于存储最近几次的损失值 - 早停法

    for epoch in progress_bar:
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        if scheduler is not None:
            scheduler.step()

        if test_loader is not None:
            assert exp != 'exp3', "exp3 does not support test while training."
            predictions, ground_truths = test(model, test_loader, device)
            accuracy = np.sum(np.array(predictions) == np.array(ground_truths)) / len(ground_truths)
            val_accuracies.append(accuracy)

            progress_bar.set_postfix({
                "Epoch": epoch + 1,
                "Loss": f"{avg_loss:.4f}",
                "Validation Acc": f"{accuracy:.4f}"
            })

            # 记录日志信息
            log_progress(epoch, avg_loss, model_name, train_time, accuracy)
        else:
            progress_bar.set_postfix({
                "Epoch": epoch + 1,
                "Loss": f"{avg_loss:.4f}"
            })

            log_progress(epoch, avg_loss, model_name, train_time)

        # 早停检查
        loss_queue.append(avg_loss)
        if len(loss_queue) == patience:
            # 检查最近几次损失的变化是否小于阈值
            if max(loss_queue) - min(loss_queue) < delta:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    # 绘制训练曲线
    # plot_training_curves(train_losses, val_accuracies, exp, model_name)

    # 保存模型权重
    weight_path = f"weights/{model_name}_weight.pth"
    torch.save(model.state_dict(), weight_path)


def plot_training_curves(train_losses, val_accuracies, exp, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'Training Curves for {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'fig/{exp}_training_{model_name}.png')
    plt.show()


def test(model, test_loader, device, weight=None, is_segmentation=False):
    """
    模型测试函数，用于分类或分割问题。
    :param model: 测试模型
    :param weight: 预训练模型权重
    :param test_loader: 测试集加载器
    :param device: 设备 cpu or cuda
    :param is_segmentation: 是否为分割问题
    """
    model.eval()
    predictions = []
    ground_truths = []

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if is_segmentation:
                predicted = torch.sigmoid(outputs)
            else:
                _, predicted = torch.max(outputs, 1)  # 获取每个样本的类别标签

            predictions.append(predicted.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

    # 将列表展平为一维数组（分类任务）
    if not is_segmentation:
        predictions = [item for sublist in predictions for item in sublist]
        ground_truths = [item for sublist in ground_truths for item in sublist]

    return np.array(predictions), np.array(ground_truths)



def plot_ga_convergence(costs, ax=None, save_fig=False, show=True):
    """
    绘制遗传算法收敛曲线
    :param costs: 每代的代价 (成本)
    :param ax: 要绘制的坐标轴，默认为 None（表示新建一个图形）
    :param save_fig: 是否保存图像
    :param show: 是否显示图像
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    x = range(len(costs))
    ax.set_title("GA Convergence")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cost (KM)')
    ax.plot(x, costs, '-', label='Cost')
    ax.text(len(x) // 2, costs[0], f'Min cost: {costs[-1]} KM', ha='center', va='center')
    ax.legend()

    if save_fig:
        plt.savefig('fig/exp5_curve.png')

    if show:
        plt.show()


def plot_route_ui(individual, ax, save_fig=False):
    """
    绘制算法求解路径，使用UI界面时不展示地图背景
    :param individual: 包含路径信息的个体
    :param ax: 来自UI的画布
    :param save_fig: 是否保存图像
    """
    lons = [gene.lng for gene in individual.genes]
    lats = [gene.lat for gene in individual.genes]
    lons.append(lons[0])
    lats.append(lats[0])

    ax.plot(lons, lats, 'ro-', label="Route")
    ax.set_title("Shortest Route")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    if save_fig:
        plt.savefig('fig/route.png')


def plot_route(individual, save_fig=False):
    """
    绘制带有中国区域地图背景和省市边界的路径图
    :param individual: 包含路径信息的个体
    :param save_fig: 是否保存图像
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())  # [min_lon, max_lon, min_lat, max_lat]
    china_shapefile_path = 'data/China_map/中华人民共和国.shp'
    gdf = gpd.read_file(china_shapefile_path)
    gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # 黑色省界

    lons = [gene.lng for gene in individual.genes]
    lats = [gene.lat for gene in individual.genes]
    route = [gene.name for gene in individual.genes] + [individual.genes[0].name]
    print("路径: ", route)

    # 补充形成闭环
    lons.append(lons[0])
    lats.append(lats[0])

    ax.plot(lons, lats, 'ro-', label="Route", markersize=5)  # 红色点和线
    ax.legend()
    ax.set_title("Shortest Route in China with Provincial Boundaries")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if save_fig:
        plt.savefig('fig/exp5_TSP_route.png', bbox_inches='tight')

    plt.show()
