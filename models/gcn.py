#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/5 下午5:47 
* Project: AI_Intro 
* File: gcn.py
* IDE: PyCharm 
* Function: GCN model
"""
import torch
from torch_geometric.nn import SplineConv, GCNConv, GATConv, ChebConv
import torch.nn.functional as F


class SplineGCN(torch.nn.Module):
    """基于B样条的图卷积网络"""
    def __init__(self, num_feature, num_classes):
        super(SplineGCN, self).__init__()
        self.conv1 = SplineConv(num_feature, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    """经典的图卷积网络"""
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """基于注意力机制的图卷积网络"""
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=3):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, 16, K)
        self.conv2 = ChebConv(16, out_channels, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
