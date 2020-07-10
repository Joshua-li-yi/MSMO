# -*- coding:utf-8 -*-
# @Time： 2020-07-09 19:22
# @Author: Joshua_yi
# @FileName: data.py
# @Software: PyCharm
# @Project: MSMO
# @Description:

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod


def get_paths(path, data_name, use_restval=False):
    """
    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    imgdir = '../data/data_preview/img'
    cap = r'dataset.json'

    roots['train'] = {'img': imgdir, 'cap': cap}
    roots['val'] = {'img': imgdir, 'cap': cap}
    roots['test'] = {'img': imgdir, 'cap': cap}
    # ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        caption = []

        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    dataset = FlickrDataset(root=root,
                            split=split,
                            json=json,
                            vocab=vocab,
                            transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    # Build Dataset Loader
    roots, _ = get_paths(dpath, data_name, opt.use_restval)

    transform = get_transform(data_name, 'train', opt)
    train_loader = get_loader_single(opt.data_name, 'train',
                                     roots['train']['img'],
                                     roots['train']['cap'],
                                     vocab, transform, ids=None,
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn)

    transform = get_transform(data_name, 'val', opt)
    val_loader = get_loader_single(opt.data_name, 'val',
                                   roots['val']['img'],
                                   roots['val']['cap'],
                                   vocab, transform, ids=None,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    # Build Dataset Loader
    roots, _ = get_paths(dpath, data_name, opt.use_restval)

    transform = get_transform(data_name, split_name, opt)
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=None,
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn)

    return test_loader

