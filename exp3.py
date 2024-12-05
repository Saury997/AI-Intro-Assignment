#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/4 上午11:23 
* Project: AI_Intro 
* File: exp3.py
* IDE: PyCharm 
* Function: Liver images segmentation using U-net
"""
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from models import UNet
from util import train, test
from loss import DiceLoss
from metric import *
from torch import optim
from dataset import LiverDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('action', type=str, default='train', help='train or test')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
parser.add_argument('--ckpt', type=str, default='weights/UNet_weight.pth', help='the path of the mode weight file')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--delta', type=float, default=1e-4, help='early stopping delta')
args = parser.parse_args()


if __name__ == '__main__':
    # 导入数据集
    train_set = LiverDataset("data/liver_dataset/train")
    test_set = LiverDataset("data/liver_dataset/val")

    # 构建数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = UNet(3, 1)

    if args.action == 'train':
        train(
            model=model,
            train_loader=train_loader,
            test_loader=None,   # 不测试
            criterion=DiceLoss(),
            optimizer=optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            ),
            scheduler=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_epochs=args.num_epochs,
            exp="exp3",
            patience=args.patience,
            delta=args.delta
        )

    elif args.action == 'test':
        model.load_state_dict(torch.load(args.ckpt))
        print(f"------Test model {model.__class__.__name__}------")

        predictions, ground_truths = test(
            model=model,
            test_loader=test_loader,
            device=torch.device("cpu"),
            weight=args.ckpt,
            is_segmentation=True
        )

        # 计算评价指标
        metric = SegmentationMetric(numClass=2)
        metrics = metric.evaluate_batch(predictions, ground_truths)

        # 输出指标
        print("Pixel Accuracy (PA): ", metrics['PA'])
        print("Class Pixel Accuracy (cPA): ", metrics['cPA'])
        print("Mean Pixel Accuracy (mPA): ", metrics['mPA'])
        print("IoU per class: ", metrics['IoU'])
        print("Mean IoU (mIoU): ", metrics['mIoU'])
        print("Frequency Weighted IoU (FWIoU): ", metrics['FWIoU'])

        # 显示部分分割效果
        num_images = min(3, len(predictions))  # 显示前三张图片
        for i in range(num_images):
            pred = predictions[i]  # 获取预测图像
            gt = ground_truths[i]  # 获取真实标签
            img = test_loader.dataset[i][0]  # 获取原图

            # 转换为灰度图
            pred = np.squeeze(pred)  # 如果是多通道，去除通道维度
            gt = np.squeeze(gt)  # 去除通道维度
            img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式

            # 绘制原图、预测结果和真实标签
            plt.figure(figsize=(12, 4))

            # 原图
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Original Image {i + 1}")
            plt.axis('off')

            # 预测结果
            plt.subplot(1, 3, 2)
            plt.imshow(pred, cmap='gray')
            plt.title(f"Prediction {i + 1}")
            plt.axis('off')

            # 真实标签
            plt.subplot(1, 3, 3)
            plt.imshow(gt, cmap='gray')
            plt.title(f"Ground Truth {i + 1}")
            plt.axis('off')

            plt.show()
