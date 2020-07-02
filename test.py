# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:57
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: MSMO
import torch
import torch.nn as nn
import os
import torchvision.transforms as tfs
from PIL import Image
import numpy as np
import datetime
import torch.nn.functional as F
import torchvision
from data_util import config
import cv2

#不是import torch.utils.data.Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo

DATA = "./data"
WIDTH = 480
HEIGHT = 320
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32


class img_dataset(Dataset):
    # 加载数据图像,train参数决定
    def loadImage(self, root=config.train_img_path, train=True):
        if train:
            images = os.listdir(config.train_img_path)
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"

        return images

    def __init__(self,train,crop_size):
        self.loadImage()

    # 将numpy数组替换为对应种类
    def image2label(self, im):
        data = np.array(im, dtype='int32')
        # print(data)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

    # 选取固定区域
    def rand_crop(self, data, label, crop_size):
        data = tfs.CenterCrop((crop_size[0], crop_size[1]))(data)
        label = tfs.CenterCrop((crop_size[0], crop_size[1]))(label)
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def img_transforms(self, im, label, crop_size):
        im, label = self.rand_crop(im, label, crop_size)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),  # [0-255]--->[0-1]
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差。Normalize之后，神经网络在训练的过程中，梯度对每一张图片的作用都是平均的，也就是不存在比例不匹配的情况
        ])
        im = im_tfs(im)
        label = self.image2label(label)
        label = torch.from_numpy(label)
        return im, label

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]


    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.img_transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)


def loadImage(root=config.train_img_path, train=True):
    if train:
        images = os.listdir(config.train_img_path)
    else:
        txt = root + "/ImageSets/Segmentation/" + "val.txt"

    return images

images = loadImage()
for i in images:
    a = Image.open(config.train_img_path + i)
    print(a)
cv2.resize(src=a, dsize=)