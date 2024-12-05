#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/4 下午4:54 
* Project: AI_Intro 
* File: metric.py
* IDE: PyCharm 
* Function: metric for exp3
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np


__all__ = ['SegmentationMetric']

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 初始化混淆矩阵

    def pixelAccuracy(self):
        # 计算像素级精确度
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # 计算每个类别的像素精确度（即每类的精度）
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        # 计算平均像素精度（MPA）
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        # 计算每个类别的IoU
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        # 计算平均IoU（mIoU）
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        imgPredict = imgPredict.astype(np.int32)
        imgLabel = imgLabel.astype(np.int32)

        # 生成混淆矩阵
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # 计算频权交并比（FWIoU）
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        if len(imgLabel.shape) == 4:  # 如果标签的维度是 (batch_size, 1, height, width)
            imgLabel = np.squeeze(imgLabel, axis=1)  # 去掉第二个维度

        if len(imgPredict.shape) == 4:  # 如果标签的维度是 (batch_size, 1, height, width)
            imgPredict = np.squeeze(imgPredict, axis=1)  # 去掉第二个维度

        # 批量添加预测和标签，更新混淆矩阵
        assert imgPredict.shape == imgLabel.shape, f"预测和标签的形状不一致: {imgPredict.shape}与{imgLabel.shape}"
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        # 重置混淆矩阵
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def evaluate_batch(self, predictions, ground_truths):
        for i in range(len(predictions)):
            self.addBatch(predictions[i], ground_truths[i])

        pa = self.pixelAccuracy()
        cpa = self.classPixelAccuracy()
        mpa = self.meanPixelAccuracy()
        IoU = self.IntersectionOverUnion()
        mIoU = self.meanIntersectionOverUnion()
        fwIoU = self.Frequency_Weighted_Intersection_over_Union()

        return {
            'PA': pa,
            'cPA': cpa,
            'mPA': mpa,
            'IoU': IoU,
            'mIoU': mIoU,
            'FWIoU': fwIoU
        }
