#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/4 上午11:29 
* Project: AI_Intro 
* File: dataset.py
* IDE: PyCharm 
* Function: Liver image dataset
"""
import os
import torch.utils.data as data
import PIL.Image as Image
from torchvision.transforms import transforms as T


class LiverDataset(data.Dataset):
    def __init__(self, root):
        n = len(os.listdir(root)) // 2

        images = []
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)
            mask = os.path.join(root, "%03d_mask.png" % i)
            images.append([img, mask])

        self.images = images
        self.transform = T.Compose([
            T.ToTensor(),
           # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.target_transform = T.ToTensor()

    def __getitem__(self, index):
        x_path, y_path = self.images[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = LiverDataset("./data/liver_dataset/train")
    print(len(dataset))
    print(dataset[0][0].dim())
