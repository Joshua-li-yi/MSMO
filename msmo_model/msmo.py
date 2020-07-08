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
from msmo_model.train_util import tensor_shape
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
        # FIXME (ly, 20200626): 这里的第几层不是很确定, 但是维度是对的
        local_features_net = net.features[:37]
        global_features_net = net.classifier[:2]
        self.global_features_net = global_features_net
        self.local_features_net = local_features_net
        # 这里不需要再次训练
        self.local_features_net.eval()
        self.global_features_net.eval()
        pass

    def forward(self, input):
        """
        :param input: 输入的batch size张图片
        :return global_features: 全局特征 4096 dimensions
                local_features: 局部特征 A = (a_1, …… ，a_L) L = 49, a_l (512) dimensions
        """

        local_features = self.local_features_net(input[0])
        # 维度转化
        local_features_output = local_features.view(-1, 2*config.hidden_dim, 49)  # B*49*(2*hidden_dim)

        global_features = self.global_features_net(local_features.view(local_features.size(0), -1))

        return local_features_output, global_features


class img_attention(nn.Module):
    """
    3 variants
    1 ATG: attention on global features
    2 ATL: attention on local features
    3 HAN: hierarchical visual attention on local features
    """

    def __init__(self, img_attention_model=None, d_h=config.hidden_dim * 2):
        super(img_attention, self).__init__()
        self.img_attention_model = img_attention_model
        global_features_dim = 4096

        if img_attention_model == 'ATG':
            self.w_g = nn.Linear(in_features=global_features_dim, out_features=global_features_dim)
            self.g_star = nn.Linear(in_features=global_features_dim, out_features=d_h)

            self.w_g_star = nn.Linear(in_features=d_h, out_features=1, bias=False)
            self.w_s_t = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs, bias=False)
            self.v = nn.Linear(in_features=config.maxinum_imgs, out_features=config.maxinum_imgs, bias=False)

        elif img_attention_model == 'ATL':
            self.w_g = nn.Linear(in_features=512, out_features=512)
            self.g_star = nn.Linear(in_features=512, out_features=d_h)

            self.w_g_star = nn.Linear(in_features=d_h, out_features=1, bias=False)
            self.w_s_t = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs * 49, bias=False)
            self.v = nn.Linear(in_features=config.maxinum_imgs * 49, out_features=config.maxinum_imgs * 49, bias=False)

        elif img_attention_model == 'HAN':
            self.g_star = nn.Linear()

    def forward(self, global_features, local_features, s_t_hat, coverage_img):
        """

        :param global_features: (tensor) (B*M)*4096
        :param local_features: (tensor) (B*M)*(2*hidden_dim)*49
        :param s_t_hat: B*(2*hidden_dim)
        :param coverage_img: B*M
        :param c_i: B*1*(2*hidden_dim)
        :param encoder_outputs:
        :return: c_img : B*1*(2*hidden_dim)
                coverage_img : B*M
                attr_img : B*M
        """
        if self.img_attention_model == 'ATG':
            """

            :param global_features: (tensor) (B*M)*4096
            :param local_features: (tensor) (B*M)*(2*hidden_dim)*49
            :param s_t_hat: B*(2*hidden_dim)
            :param coverage_img: B*M
            :param c_i: B*1*(2*hidden_dim)
            :param encoder_outputs:
            :return: c_img : B*1*(2*hidden_dim)
                    coverage_img : B*M
                    attr_img : B*M
            """

            g_star = self.w_g(global_features)  # M*4096
            g_star = self.g_star(g_star)  # M*(2*hidden_dim)
            w_g_star = self.w_g_star(g_star)  # M*B
            w_g_star = w_g_star.permute(1, 0).contiguous()   # B*M
            w_s_t = self.w_s_t(s_t_hat)  # B*M
            # F.tanh改为了 torch.tanh
            e_a = torch.tanh(w_g_star + w_s_t + coverage_img)  # B*M
            e_a = self.v(e_a)  # B*M
            attr_img = torch.softmax(e_a, dim=1)  # B*M
            attr_img = attr_img.unsqueeze(1)  # B x 1 x M

            g_star = g_star.unsqueeze(1)  # M*B*(2*hidden_dim)
            g_star = g_star.permute(1, 0, 2).contiguous()  # B*M*(2*hidden_dim)
            # 矩阵相乘
            c_img = torch.bmm(attr_img, g_star)  # B x 1 x n
            c_img = c_img.view(-1, config.hidden_dim * 2).contiguous()  # B*(2*hidden_dim)
            attr_img = attr_img.squeeze(1).contiguous()  # B*M
            coverage_img = coverage_img.squeeze(1).contiguous()  # B*M

            coverage_img = coverage_img + attr_img
            return c_img, attr_img, coverage_img
        elif self.img_attention_model == 'ATL':
            """
            
            :param global_features: (tensor) (B*M)*4096
            :param local_features: (tensor) (B*M)*512*49
            :param s_t_hat: B*(2*hidden_dim)
            :param coverage_img: B*(M*49)
            :param c_i: B*1*(2*hidden_dim)
            :return: c_img : B*(2*hidden_dim)
                    coverage_img : B*(49*M)
                    attr_img : B*(49*M)
            """
            # (49*M) * 512
            local_features = local_features.view(-1, 512)

            g_star = self.w_g(local_features)  # (49*M)*512

            g_star = self.g_star(g_star)  # (49*M)*(2*hidden_dim)

            w_g_star = self.w_g_star(g_star)  # (49*M)*1

            w_g_star = w_g_star.permute(1, 0).contiguous()  # 1*(49*M)

            w_s_t = self.w_s_t(s_t_hat)  # 1*(49*M)

            # F.tanh改为了 torch.tanh
            e_a = torch.tanh(w_g_star + w_s_t + coverage_img)  # 1*(49*M)
            e_a = self.v(e_a)   # 1*(49*M)
            tensor_shape(e_a)  # 1*(49*M)
            attr_img = torch.softmax(e_a, dim=1)  # 1*(49*M)

            g_star = g_star.unsqueeze(1)  # (49*M)*1*(2*hidden_dim)
            g_star = g_star.permute(1, 0, 2).contiguous()  # 1*(49*M)*(2*hidden_dim)

            attr_img = attr_img.unsqueeze(1).contiguous()  # 1*1*(49*M)

            # 矩阵相乘
            c_img = torch.bmm(attr_img, g_star)  # 1*1*(2*hidden_dim)

            c_img = c_img.view(-1, config.hidden_dim * 2).contiguous()  # B*(2*hidden_dim)
            attr_img = attr_img.squeeze(1).contiguous()  # B*M
            coverage_img = coverage_img.squeeze(1).contiguous()  # B*M

            coverage_img = coverage_img + attr_img
            return c_img, attr_img, coverage_img

        elif self.img_attention_model == 'HAN':
            g_star = self.g_star(global_features)
            e_a = self.e_a(g_star, s_t_hat)
            e_a = torch.tanh(e_a)
            alpha_a = torch.softmax(e_a)





class multi_attention(nn.Module):
    def __init__(self):
        super(multi_attention, self).__init__()
        self.w_c_txt = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.hidden_dim * 2, bias=False)
        self.w_s_txt = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.hidden_dim * 2, bias=False)

        self.w_c_img = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.hidden_dim * 2, bias=False)
        self.w_s_img = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.hidden_dim * 2, bias=False)

        self.v_txt = nn.Linear(in_features=config.hidden_dim * 2, out_features=1, bias=False)
        self.v_img = nn.Linear(in_features=config.hidden_dim * 2, out_features=1, bias=False)

    def forward(self, input_c_txt, input_c_img, input_s_t):
        """

        :param input_c_txt: B*(2*hidden_dim)
        :param input_c_img: B*(2*hidden_dim)
        :param input_s_t: B*(2*hidden_dim)
        :return: c_mm B*(2*hidden_dim)
        """

        w_c_txt = self.w_c_txt(input_c_txt)  # B*(2*hidden_dim)
        w_s_txt = self.w_s_txt(input_s_t)  # B*(2*hidden_dim)

        w_c_img = self.w_c_img(input_c_img)  # B*(2*hidden_dim)
        w_s_img = self.w_s_img(input_s_t)  # B*(2*hidden_dim)

        e_txt = self.v_txt(w_c_txt+w_s_txt)  # B*1
        e_img = self.v_img(w_c_img+w_s_img)  # B*1
        # need dim arg
        alpha_txt = torch.softmax(e_txt, dim=1)  # B*1
        alpha_img = torch.softmax(e_img, dim=1)  # B*1

        c_mm = torch.mul(alpha_txt, input_c_txt) + torch.mul(alpha_img, input_c_img)  # B*(2*hidden_dim)
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
        # 1*400*(2*hidden_dim)
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

        # 1*1*400
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
        self.txt_attention = txt_attention()
        # 需要被训练
        self.img_attention = img_attention(img_attention_model=config.img_attention_model)
        # 需要被训练
        self.multi_attention = multi_attention()
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
                c_mm, extra_zeros, enc_batch_extend_vocab, coverage, step, golbal_features, local_features,
                coverage_img):
        # 如果不是模型不是处于训练阶段
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.txt_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                       enc_padding_mask, coverage)

            coverage = coverage_next


        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_mm, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        # 横向拼接
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_next = self.txt_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
        # 1*1*(2*hidden_dim)
        # tensor_shape(c_t)
        # 1*(2*hidden_dim)
        # tensor_shape(s_t_hat)
        # 1*400
        # tensor_shape(attn_dist)
        # 1*400
        # tensor_shape(coverage_next)

        c_i,  attn_img, coverage_img_next = self.img_attention(golbal_features, local_features, s_t_hat, coverage_img)
        # 1*(2*hidden_dim)
        # tensor_shape(c_i)
        # 1*10
        # tensor_shape(coverage_img_next)
        # 1*10
        # tensor_shape(attn_img)
        c_mm = self.multi_attention(c_t, c_i, s_t_hat)
        # tensor_shape(c_mm)

        # 模型处于训练阶段
        if self.training or step > 0:
            coverage = coverage_next
            coverage_img = coverage_img_next

        p_gen = None
        if config.pointer_gen:
            # p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            # 多模态
            p_gen_input = torch.cat((c_mm, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        # output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_mm), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hid   den_dim

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
        return final_dist, s_t, c_mm, attn_dist, p_gen, coverage, attn_img, coverage_img
        # TODO(20200703): 加最后的选择图片部分


class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        txt_encode = txt_encoder()
        reduce_state = ReduceState()
        img_encode = img_encoder()
        decoder = Decoder()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = txt_encode.embedding.weight
        # only forward calculate
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

        # load model state
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.txt_encode.load_state_dict(state['txt_encode_state_dict'])
            self.img_encode.load_state_dict(state['img_encode_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
