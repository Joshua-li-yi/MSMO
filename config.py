# -*- coding:utf-8 -*-
# @Time： 2020-06-24 11:11
# @Author: Joshua_yi
# @FileName: config.py
# @Software: PyCharm
# @Project: MSMO

import os

root_dir = r'../'

train_data_path_ATG = os.path.join(root_dir, r"data/data_preview/msmo_data/train/train_ATG.json")
train_data_path_ATL_HAN = os.path.join(root_dir, r"data/data_preview/msmo_data/train/train_ATL_HAN.json")

valid_data_path_ATG = os.path.join(root_dir, r"data/data_preview/msmo_data/valid/valid_ATG.json")
valid_data_path_ATL_HAN = os.path.join(root_dir, r"data/data_preview/msmo_data/valid/valid_ATL_HAN.json")

test_data_path_ATG = os.path.join(root_dir, r"data/data_preview/msmo_data/test/test_ATG.json")
test_data_path_ATL_HAN = os.path.join(root_dir, r"data/data_preview/msmo_data/test/test_ATL_HAN.json")

# vocab的路径
vocab_path = os.path.join(root_dir, r"data/data_preview/msmo_data/vocab.txt")
word_id_path = os.path.join(root_dir,r'data/data_preview/word_id.csv')

msmo_logging_path = os.path.join('', r"logging.log")

# Hyperparameters
hidden_dim = 256
emb_dim = 128

batch_size = 1

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
model_maxinum_imgs = {
    'ATG': 10,
    'ATL': 7,
    'HAN': 7
}
# 选择所用的模型
img_attention_model = img_attention_models[1]
maxinum_imgs = model_maxinum_imgs[img_attention_model]

train_data_path = [train_data_path_ATL_HAN, train_data_path_ATG][img_attention_model == 'ATG']
msmo_modle_path = os.path.join(root_dir, r'msmo_model/')

#################### MMAE ##################3
mmae_methods = ['LR', 'Logis', 'MLP']
mmae_method = mmae_methods[0]
mmae_model_path = os.path.join(root_dir, r'mmae_model/')

