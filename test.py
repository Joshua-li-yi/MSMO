# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:57
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: MSMO
# Owed by: http://blog.csdn.net/chunyexiyu
# direct get the input name from called function code


# 导入相关模块
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np


class AnimalData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = io.imread(img_path)  # 读取该图片
        label = img_path.split('\\')[-1].split('.')[
            0]  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本

