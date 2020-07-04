# -*- coding:utf-8 -*-
# @Time： 2020-06-24 13:52
# @Author: Joshua_yi
# @FileName: train_util.py
# @Software: PyCharm
# @Project: MSMO

from torch.autograd import Variable
import numpy as np
import torch
from data_util import config
import time
import torch
# 时间使用装饰器
# 使用时直接在函数前加 @timer

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        print(f"Cost time: {cost} s")
        return r
    return wrapper

import inspect
def retrieve_name_ex(var):
    stacks = inspect.stack()
    try:
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc)
        startIndex = code.index("(", startIndex + len(callFunc)) + 1
        endIndex = code.index(")", startIndex)
        return code[startIndex:endIndex].strip()
    except:
        return ""


def tensor_shape(value):
    print("{} shape:  {}".format(retrieve_name_ex(value), value.shape))
    pass


def get_input_from_batch(batch, use_cuda):

    batch_size = len(batch.enc_lens)
    imgs = batch.original_imgs

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    # 上下文向量初始化为0
    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
    c_i = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
    c_i = c_i.unsqueeze(0)
    # coverage初始化为0
    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))
        coverage_img = Variable(torch.zeros(config.batch_size, enc_batch.size(),2*config.hidden_dim))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, imgs, c_i, coverage_img


def get_output_from_batch(batch, use_cuda):
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
