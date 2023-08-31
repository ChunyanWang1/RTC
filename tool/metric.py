#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from ptflops.flops_counter import get_model_complexity_info


####################
# pixel accuracy 所有像素点中预测正确的点的比率
# mean accuraccy 对所有类别的accuracy取平均
# mean IU
# frequency weighted IU
####################
def _fast_hist(label_true, label_pred, n_class):
    """计算混淆矩阵"""
    mask = (label_true >= 0) & (label_true < n_class) # 忽略掉-1类或255类，这两个类用于表示pascal voc图片中的多余轮廓线
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """label_trues: Numpy array, [m, n]
    label_preds: Numpy array, [m, n]
    Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    #print("metric.py", label_trues.shape, label_preds.shape)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)  # 混淆矩阵
    #print('metric.py,hist', hist.sum(axis=1))
    acc = np.diag(hist).sum() / hist.sum()  # 混淆矩阵的对角线上是预测正确的像素点个数
    acc_cls = np.diag(hist) / hist.sum(axis=1)  # 混淆矩阵的一行元素之和，表示属于该类的像素点个数
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def label_accuracy_score_perclass(label_trues, label_preds, n_class):
    """label_trues: Numpy array, [m, n]
    label_preds: Numpy array, [m, n]
    Returns accuracy score evaluation result when valiating or testing.
    Notice that if there is NAN, the average results will be uncorrect.
      - pixel accuracy per class
      - IoU per class
      - fwavacc per class
    """
    #print("metric.py", label_trues.shape, label_preds.shape)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)  # 混淆矩阵
    #print('metric.py,hist', hist.sum(axis=1))
    acc_per_cls = np.diag(hist) / hist.sum(axis=1)  # 混淆矩阵的一行元素之和，表示属于该类的像素点个数
    iou_per_cls = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc_per_cls = freq * iou_per_cls
    #print(acc_per_cls, acc_per_cls.shape)
    #print(iou_per_cls, iou_per_cls.shape)
    #print(fwavacc_per_cls, fwavacc_per_cls.shape)

    return acc_per_cls, iou_per_cls, fwavacc_per_cls


def label_accuracy_score_forTrain(label_trues_list, label_preds_list, n_class):
    """label_trues: Numpy array, [b,m,n]
    label_preds: Numpy array, [b,m,n]
    Returns accuracy score evaluation result when training.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    #print("metric.py", label_trues_list.shape, label_preds_list.shape)
    acc, acc_cls, mean_iu, fwavacc = [], [], [], []
    for label_trues, label_preds in zip(label_trues_list, label_preds_list):
        evals = label_accuracy_score(label_trues, label_preds, n_class)
        acc.append(evals[0])
        acc_cls.append(evals[1])
        mean_iu.append(evals[2])
        fwavacc.append(evals[3])
    avg_acc = sum(acc) / len(acc)
    avg_acc_cls = sum(acc_cls) / len(acc_cls)
    avg_mean_iu = sum(mean_iu) / len(mean_iu)
    avg_fwavacc = sum(fwavacc) / len(fwavacc)

    return avg_acc, avg_acc_cls, avg_mean_iu, avg_fwavacc



####################
# 剔除背景像素点的计算
# pixel accuracy 所有像素点中预测正确的点的比率
# mean accuraccy 对所有类别的accuracy取平均
# mean IU
# frequency weighted IU
####################
def _fast_hist_nobackg(label_true, label_pred, n_class):
    """计算混淆矩阵"""
    mask = (label_true > 0) & (label_true < n_class)  # 忽略掉-1类或255类，这两个类用于表示pascal voc图片中的多余轮廓线
                                                      # 忽略掉0类，即背景类
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score_nobackg(label_trues, label_preds, n_class):
    """label_trues: Numpy array, [m, n]
    label_preds: Numpy array, [m, n]
    Returns accuracy score evaluation result when valiating or testing.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    #print("metric.py", label_trues.shape, label_preds.shape)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist_nobackg(lt.flatten(), lp.flatten(), n_class)  # 混淆矩阵
    #print('metric.py,hist', hist.sum(axis=1))
    acc = np.diag(hist).sum() / hist.sum()  # 混淆矩阵的对角线上是预测正确的像素点个数
    acc_cls = np.diag(hist) / hist.sum(axis=1)  # 混淆矩阵的一行元素之和，表示属于该类的像素点个数
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



####################
# flops
# params
####################
def cal_flops_params(model):
    '''Notice, as for my file structure, the model refer to xxx_arch
    '''
    
    flops, para = get_model_complexity_info(model, (3,128,128), True, True)
    print('Flops', flops)
    print('para', para)
