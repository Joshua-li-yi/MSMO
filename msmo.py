# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:56
# @Author: Joshua_yi
# @FileName: msmo.py
# @Software: PyCharm
# @Project: MSMO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO (ly, 20200622): parameters
train_dir = "data/train"           #训练集路径
# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# TODO (ly, 20200622): load data
# train_dataset = torchvision.datasets.MNIST(root='/data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
# test_dataset = torchvision.datasets.MNIST(root='/data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# TODO (ly, 20200622): text pretreatment


# TODO (ly, 20200622): image encode
class VGGNet(nn.Module):
    def __init__(self, num_classes=3):
        super(VGGNet, self).__init__()
        # 导入模型结构
        net = models.vgg19(pretrained=False)
        net.load_state_dict(torch.load(r'./model/vgg19.pth'))
        # 加载预先下载好的预训练参数到resnet18
        print(type(net))

        # summary(net, input_size=(3, 225, 225))
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------训练过程---------------------------------
model = VGGNet().to(device)
# params = [{'params': md.parameters()} for md in model.children()
#           if md in [model.classifier]]

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
print(model)
Loss_list = []
Accuracy_list = []

# TODO (ly, 20200622): text encode

# TODO (ly, 20200622): text attention

# TODO (ly, 20200622): image attention

# TODO (ly, 20200622): multimodal attention

# TODO (ly, 20200622): decode

# TODO (ly, 20200622): train model

# for epoch in range(100):
#     print('epoch {}'.format(epoch + 1))
#     # training-----------------------------
#     train_loss = 0.
#     train_acc = 0.
#     for batch_x, batch_y in train_dataloader:
#         batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
#         out = model(batch_x)
#         loss = loss_func(out, batch_y)
#         train_loss += loss.data[0]
#         pred = torch.max(out, 1)[1]
#         train_correct = (pred == batch_y).sum()
#         train_acc += train_correct.data[0]
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
#         train_datasets)), train_acc / (len(train_datasets))))
#
#     # evaluation--------------------------------
#     model.eval()
#     eval_loss = 0.
#     eval_acc = 0.
#     for batch_x, batch_y in val_dataloader:
#         batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
#         out = model(batch_x)
#         loss = loss_func(out, batch_y)
#         eval_loss += loss.data[0]
#         pred = torch.max(out, 1)[1]
#         num_correct = (pred == batch_y).sum()
#         eval_acc += num_correct.data[0]
#     print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#         val_datasets)), eval_acc / (len(val_datasets))))
#
#     Loss_list.append(eval_loss / (len(val_datasets)))
# Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
#
# x1 = range(0, 100)
# x2 = range(0, 100)
# y1 = Accuracy_list
# y2 = Loss_list
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.show()

# TODO (ly, 20200622): test