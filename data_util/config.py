# -*- coding:utf-8 -*-
# @Time： 2020-06-24 11:11
# @Author: Joshua_yi
# @FileName: config.py
# @Software: PyCharm
# @Project: MSMO

import os

root_dir = r'D:/githubProject/MSMO/'

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, r"data/data_preview/finished_files/train.bin")
eval_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/test.bin")
# vocab的路径
vocab_path = os.path.join(root_dir, r"data/data_preview/finished_files/vocab.txt")
word_id_path = os.path.join(root_dir,r'data/data_preview/word_id.csv')

log_root = os.path.join(root_dir, "log")

train_caption_path = os.path.join(root_dir, r'data/data_preview/caption/')
train_img_path = os.path.join(root_dir, r'data/data_preview/img/')
train_article_path = os.path.join(root_dir, r'data/data_preview/article/')
train_url_path = os.path.join(root_dir, r'data/data_preview/url/')

# Hyperparameters
hidden_dim = 256
emb_dim = 128

batch_size = 8

max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 50000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
# 是否使用指针生成，从PGN net code 改写过来的，保留此选项，始终为True即可
pointer_gen = True
# 是否覆盖
is_coverage = True

cov_loss_wt = 1.0

eps = 1e-12
# 训练的最大迭代次数
max_iterations = 500000
# 是否使用cuda加速
use_gpu = False

lr_coverage = 0.15
# 随机种子
SEED = 123
# 图片的注意力机制选择
img_attention_models = ['ATG', 'ATL', 'HAN']
# 选择所用的模型
img_attention_model = img_attention_models[0]
# MMAE模型的选择
methods = ['LR', 'Logis', 'MLP']
method = methods[0]
modle_path = ''