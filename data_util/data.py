# -*- coding:utf-8 -*-
# @Time： 2020-06-24 11:10
# @Author: Joshua_yi
# @FileName: data.py
# @Software: PyCharm
# @Project: MSMO

import glob
import random
import struct
import csv
import ujson
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.
class Vocab(object):

    def __init__(self, vocab_file, max_size):
        """
        :param vocab_file: 读取词汇文件的路径
        :param max_size: 最大词汇数
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                # print(pieces)
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                w = pieces[0]

                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)

                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1

                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        """

        :param fpath: 写入词汇文件的路径
        :return:
        """
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w", encoding='utf-8') as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)

            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass=True):
    """
    生成器
    :param data_path: 读取数据的路径
    :param single_pass: 是否按顺序读取文件，默认为True
    :return:
    """
    with open(data_path, 'r') as f:
        train_data = ujson.load(f)
    for train in train_data:
        print(train['article'], train['abstract'], train['imgs'])

        yield train['article'], train['abstract'], train['imgs']

    # while True:
    #     filelist = glob.glob(data_path)  # get the list of datafiles
    #     assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
    #
    #     if single_pass:
    #         filelist = sorted(filelist)
    #     else:
    #         random.shuffle(filelist)
    #
    #
    #     # TODO(ly, 20200630): 读懂数据读取这一块儿
    #     for f in filelist:
    #         reader = open(f, 'rb')
    #         while True:
    #             len_bytes = reader.read(8)
    #             if not len_bytes: break  # finished reading this file
    #             str_len = struct.unpack('q', len_bytes)[0]
    #             example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    #
    #             yield example_pb2.Example.FromString(example_str)
    #     else:
    #         print("example_generator completed reading all datafiles. No more data.")
    #
    #     if single_pass:
    #         print("example_generator completed reading all datafiles. No more data.")
    #         break


def article2ids(article_words, vocab):
    ids = []
    oovs = []  # out-of-vocabulary
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            # print(SENTENCE_START)
            # print(type(SENTENCE_START))
            # print(abstract)
            abstract = str(abstract)
            # print(abstract)
            # print(cur)
            # print(type(cur))
            # 查找<s> </s>首次出现的位置
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str

# ------------------------------MMAE-----------------------
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


#不是import torch.utils.data.Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo

DATA = "./data"
WIDTH = 480
HEIGHT = 320
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32


class VOCSegDataSet(Dataset):
    # 加载数据图像,train参数决定
    def loadImage(self, root=config.train_img_path, train=True):
        if train:
            images = os.listdir(config.train_img_path)
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"

        return images

    def __init__(self,train,crop_size):
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'potted plant',
                        'sheep', 'sofa', 'train', 'tv/monitor']

        # 种类对应的RGB值
        self.colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                         [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        # 将RGB值映射为一个数值
        self.cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(self.colormap):
            self.cm2lbl[cm[0] * 256 * 256 + cm[1] * 256 + cm[2]] = i
        self.crop_size = crop_size
        data_list,label_list = self.loadImage(train = train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

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


class FCN(nn.Module):
    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)


    def __init__(self,num_classes):
        super(FCN,self).__init__()

        # self.pretrained_net = model_zoo.resnet34(pretrained=True)
        self.pretrained_net = torchvision.models.resnet34(pretrained = True)
        self.stage1 = nn.Sequential(*list(self.pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(self.pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(self.pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 8, 4, 2, bias=False)
        self.upsample_4x.weight.data = self.bilinear_kernel(num_classes, num_classes, 8)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16

        x = self.stage3(x)
        s3 = x  # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_2x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s


class Main():
    def __init__(self):
        #种类
        self.classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
        #种类对应的RGB值
        self.colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]


        self.num_classes = len(self.classes)


        #将RGB值映射为一个数值
        self.cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            self.cm2lbl[cm[0] * 256 * 256 + cm[1] * 256 + cm[2]] = i

    def loadImage(self, root=DATA, train=True):

        if train:
            txt = root + "/ImageSets/Segmentation/" + "train.txt"
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"
        with open(txt, 'r') as f:
            images = f.read().split()

        data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
        return data, label

    #将numpy数组替换为对应种类
    def image2label(self,im):
        data = np.array(im, dtype='int32')
        # print(data)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵




    #选取固定区域
    def rand_crop(self,data,label,height,width):
        data = tfs.CenterCrop((height, width))(data)
        data.show()
        label = tfs.CenterCrop((height, width))(label)
        label.show()
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def img_transforms(self,im,label,height,width):
        im,label = self.rand_crop(im,label,height,width)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),#[0-255]--->[0-1]
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值方差
        ])
        im = im_tfs(im)
        label = self.image2label(label)
        label = torch.from_numpy(label)
        return im,label

    def _fast_hist(self,label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def label_accuracy_score(self,label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc




    def main(self):
        crop_size = [HEIGHT,WIDTH]
        voc_train = VOCSegDataSet(train = True,crop_size = crop_size)
        voc_test = VOCSegDataSet(train = False,crop_size = crop_size)
        train_data = DataLoader(voc_train, TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
        valid_data = DataLoader(voc_test, VALID_BATCH_SIZE, num_workers=4)
        self.net = FCN(len(self.classes))
        self.net.cuda()
        criterion = nn.NLLLoss2d()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        for e in range(80):
            if e > 0 and e % 50 == 0:
                optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
            train_loss = 0
            train_acc = 0
            train_acc_cls = 0
            train_mean_iu = 0
            train_fwavacc = 0

            prev_time = datetime.datetime.now()
            net = self.net.train()
            for data in train_data:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)  # (b, n, h, w)
                loss = criterion(out, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = self.label_accuracy_score(lbt, lbp, self.num_classes)
                    train_acc += acc
                    train_acc_cls += acc_cls
                    train_mean_iu += mean_iu
                    train_fwavacc += fwavacc

            net = net.eval()
            eval_loss = 0
            eval_acc = 0
            eval_acc_cls = 0
            eval_mean_iu = 0
            eval_fwavacc = 0
            for data in valid_data:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.data[0]

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = self.label_accuracy_score(lbt, lbp, self.num_classes)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
                e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
                   eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
            time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
            print(epoch_str + time_str + ' lr: {}'.format(optimizer.learning_rate))

if __name__ == "__main__":
    t = Main()
    t.main()

