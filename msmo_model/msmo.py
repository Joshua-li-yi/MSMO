# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:56
# @Author: liyi
# @FileName: msmo.py
# @Software: PyCharm
# @Project: MSMO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# 导入配置
from data_util import config
from numpy import random

# 是否使用cuda加速
use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(config.SEED)
# 为CPU设置种子用于生成随机数，以十结果是确定的
torch.manual_seed(config.SEED)

# 为当前的GPU设置随机种子
# torch.cuda.manual_seed(123)

if torch.cuda.is_available():
    # 如果使用多个GPU，为所有的GPU设置种子
    torch.cuda.manual_seed_all(config.SEED)


# 初始化lstm的权重
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


# 正太分布初始化
def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


# 均匀分布初始化
def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class txt_encoder(nn.Module):

    """
    txt encode
    Attributes:
        module
    """

    def __init__(self):
        super(txt_encoder, self).__init__()
        # word embedding matrix vocab_size * emb_dim
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        # embedding层正太分布初始化
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # 自定义lstm初始化方案
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        # 转化为内存连续的tensor
        encoder_outputs = encoder_outputs.contiguous()
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class img_encoder(nn.Module):
    """
    img encode
    """
    def __init__(self):
        super(img_encoder, self).__init__()
        # 导入模型结构
        print('load vgg19...')
        net = models.vgg19(pretrained=False)
        net.load_state_dict(torch.load(r'vgg19.pth'))
        # FIXME (ly, 20200626): 这里的第几层不是很确定
        local_features_net = list(net.children())[:-9]
        global_features_net = list(net.children())[-9:-6]
        self.global_features_net = nn.Sequential(*global_features_net)
        self.local_features_net = nn.Sequential(*local_features_net)

    def forward(self, input):
        """
        # FIXME (ly, 20200701): 是输入的batch size张图片还是一张？？向量是归一化之后的还是生图？？
        :param input: 输入一张图片
        :return global_features: 全局特征 4096 dimensions
                local_features: 局部特征 A = (a_1, …… ，a_L) L = 49, a_l 512 dimensions
        """
        print()

        local_features = self.local_features_net(input)
        print("local_features_net size")

        # 维度转化
        local_features = local_features.view(-1, 521, 49)  # B*49*512
        global_features = self.global_features_net(local_features)

        return local_features, global_features


class img_attention(nn.Module):
    """
    3 variants
    1 ATG: attention on global features
    2 ATL: attention on local features
    3 HAN: hierarchical visual attention on local features
    """
    def __init__(self, img_attention_model=None, s_t_dim=0, cov_a_dim=0, d_h=0):
        super(img_attention, self).__init__()
        self.img_attention_model = img_attention_model
        global_features_dim = 4096
        if img_attention_model == 'ATG':
            self.w_g = nn.Linear(in_features=global_features_dim, out_features=global_features_dim)
            self.w_g_star = nn.Linear(in_features=global_features_dim, out_features=config.hidden_dim)

            self.g_star = nn.Sequential(
                self.w_g,
                self.w_g_star,
            )
            self.w_a = nn.Linear(in_features=config.hidden_dim, out_features=0)
            self.e_a = nn.Sequential(
                nn.Linear(in_features=d_h, out_features=0,bias=False),
                nn.Linear(in_features=s_t_dim, out_features=0,bias=False),
            )

        elif img_attention_model == 'ATL':
            self.g_star = nn.Linear()

        elif img_attention_model == 'HAN':
            self.g_star = nn.Linear()

    def forward(self, global_features, local_features, s_t, cov_a):
        if self.img_attention_model == 'ATG':
            g_star = self.g_star(global_features)
            e_a = self.e_a(g_star, s_t, cov_a)
            # F.tanh改为了 torch.tanh
            e_a = torch.tanh(e_a)
            alpha_a = torch.softmax(e_a)
        elif self.img_attention_model == 'ATL':
            g_star = self.g_star(global_features)
            e_a = self.e_a(g_star, s_t, cov_a)
            e_a = torch.tanh(e_a)

            alpha_a = torch.softmax(e_a)
        elif self.img_attention_model == 'HAN':
            g_star = self.g_star(global_features)
            e_a = self.e_a(g_star, s_t, cov_a)
            e_a = torch.tanh(e_a)
            alpha_a = torch.softmax(e_a)
        c_img = torch.sum(torch.mul(alpha_a, g_star))
        return c_img


class multi_attention(nn.Module):
    def __init__(self, c_txt_dim=0, c_img_dim=0, s_t_dim=0):
        super(multi_attention, self).__init__()
        self.s_t = nn.Linear(in_features=s_t_dim, bias=False)
        self.c_txt = nn.Linear(in_features=c_txt_dim, bias=False)
        self.e_txt = nn.Linear(in_features=c_txt_dim,bias=False)

        self.c_img = nn.Linear(in_features=c_img_dim, bias=False)
        self.e_img = nn.Linear(in_features=c_img_dim, bias=False)

    def forward(self, input_c_txt, input_c_img, input_s_t):
        s_t = self.s_t(input_s_t)
        c_txt = self.c_txt(input_c_txt)
        e_txt = self.e_txt(torch.sum(s_t, c_txt))
        a_txt = torch.softmax(e_txt)

        c_img = self.c_img(input_c_img)
        e_img = self.e_img(torch.sum(s_t, c_img))
        a_img = torch.softmax(e_img)

        c_mm = torch.sum(torch.mul(a_txt,c_txt), torch.mul(a_img, c_img))

        return c_mm


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):

        h, c = hidden  # h, c dim = 2 x b x hidden_dim

        """
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, head * d_k)
        输入的x的形状为[batch size, head, 句子最大长度max_len, d_k]，
        先执行x.transpose(1, 2)后变换为[batch size, 句子最大长度max_len, head, d_k]，
        然后因为先执行transpose后执行view的话，两者中间先要执行contiguous，
        把经过了transpose或t()操作的tensor重新处理为具有内存连续的有相同数据的tensor，
        最后才能执行view(batch_size, -1, head * d_k) 把 [batch size, 句子最大长度max_len, head, d_k]
        变换为 [batch size, 句子最大长度max_len, embedding_dim词向量维度]，head * d_k 等于 embedding_dim词向量维度。
        """

        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class txt_attention(nn.Module):
    def __init__(self):
        super(txt_attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim

        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = torch.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        # 矩阵相乘
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = txt_attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        # 横向拼接
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = torch.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        txt_encode = txt_encoder()
        reduce_state = ReduceState()
        img_encode = img_encoder()
        decoder = Decoder()


        # shared the embedding between encoder and decoder
        decoder.embedding.weight = txt_encode.embedding.weight
        # 不进行梯度回传，只进行向前计算
        if is_eval:
            txt_encode = txt_encode.eval()
            reduce_state = reduce_state.eval()
            img_encode = img_encode.eval()
            decoder = decoder.eval()


        if use_cuda:
            txt_encode = txt_encode.cuda()
            img_encode = img_encode.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.txt_encode = txt_encode
        self.img_encode = img_encode
        self.decoder = decoder
        self.reduce_state = reduce_state
        # 如果存在模型路径的话，就直接导入模型
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.txt_encode.load_state_dict(state['txt_encode_state_dict'])
            self.img_encode.load_state_dict(state['img_encode_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])