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
import config
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
        local_features = self.local_features_net(input)
        # 维度转化
        local_features_output = local_features.view(-1, 2 * config.hidden_dim, 49)  # B*49*(2*hidden_dim)

        global_features = self.global_features_net(local_features.view(local_features.size(0), -1))
        global_features = global_features.view(config.batch_size, config.maxinum_imgs, 4096)
        local_features_output = local_features_output.view(config.batch_size, config.maxinum_imgs, 512, 49)
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
        self.d_h = d_h
        self.global_features_dim = 4096
        model_dict = {'ATG': self.init_ATG, 'ATL': self.init_ATL, 'HAN': self.init_HAN}
        model_dict[img_attention_model]()
        pass

    def init_ATG(self):
        """
        init ATG attention model
        :return:
        """
        self.w_g = nn.Linear(in_features=self.global_features_dim, out_features=self.global_features_dim)
        self.g_star = nn.Linear(in_features=self.global_features_dim, out_features=self.d_h)
        self.w_g_star = nn.Linear(in_features=self.d_h, out_features=1, bias=False)
        self.w_s_t = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs, bias=False)
        self.v = nn.Linear(in_features=config.maxinum_imgs, out_features=config.maxinum_imgs, bias=False)
        pass

    def init_ATL(self):
        """
        init ATL attention model
        :return:
        """
        self.w_g = nn.Linear(in_features=512, out_features=512)
        self.g_star = nn.Linear(in_features=512, out_features=self.d_h)

        self.w_g_star = nn.Linear(in_features=self.d_h, out_features=1, bias=False)
        self.w_s_t = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs * 49, bias=False)
        self.v = nn.Linear(in_features=config.maxinum_imgs * 49, out_features=config.maxinum_imgs * 49, bias=False)
        pass

    def init_HAN(self):
        """
        init HAN attention model
        :return:
        """
        self.w_g = nn.Linear(in_features=49, out_features=self.d_h)
        self.g_star = nn.Linear(in_features=self.d_h, out_features=1)

        self.w_g_star = nn.Linear(in_features=512, out_features=512, bias=False)
        self.w_s_t = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs * 512, bias=False)
        self.v = nn.Linear(in_features=512, out_features=512, bias=False)

        self.w_g_star_i = nn.Linear(in_features=512, out_features=1, bias=False)
        self.w_s_t_i = nn.Linear(in_features=config.hidden_dim * 2, out_features=config.maxinum_imgs, bias=False)
        self.v_i = nn.Linear(in_features=config.maxinum_imgs, out_features=config.maxinum_imgs, bias=False)

        pass

    def ATG_forward(self, global_features, local_features, s_t_hat, coverage_img, coverage_patches=None):
        """
        :param global_features: ) B*M*4096
        :param s_t_hat: B*(2*hidden_dim)
        :param coverage_img: B*M
        :return: c_img : B*(2*hidden_dim)
                 coverage_img : B*M
                 attr_img : B*M
        """
        g_star = self.w_g(global_features)  # B*M*4096
        g_star = self.g_star(g_star)  # B*M*(2*hidden_dim)
        w_g_star = self.w_g_star(g_star)  # B*M*1
        w_g_star = w_g_star.squeeze(2)
        w_s_t = self.w_s_t(s_t_hat)  # B*M
        # F.tanh改为了 torch.tanh
        e_a = torch.tanh(w_g_star + w_s_t + coverage_img)  # B*M
        e_a = self.v(e_a)  # B*M
        attr_img = torch.softmax(e_a, dim=1)  # B*M
        attr_img = attr_img.unsqueeze(1)  # B x 1 x M
        c_img = torch.bmm(attr_img, g_star)  # B x 1 x n
        c_img = c_img.view(-1, config.hidden_dim * 2).contiguous()  # B*(2*hidden_dim)
        attr_img = attr_img.squeeze(1).contiguous()  # B*M

        coverage_img = coverage_img + attr_img
        return c_img, attr_img, coverage_img, ()

    def ATL_forward(self, global_features, local_features, s_t_hat, coverage_img, coverage_patches=None):
        """
        :param local_features: (tensor) B*M*512*49
        :param s_t_hat: B*(2*hidden_dim)
        :param coverage_img: B*(M*49)
        :return: c_img : B*(2*hidden_dim)
                coverage_img : B*(49*M)
                attr_img : B*(49*M)
        """
        # B*(49*M) * 512
        local_features = local_features.view(config.batch_size, -1, 512)

        g_star = self.w_g(local_features)   # B*(49*M)*512
        g_star = self.g_star(g_star)  # B*(49*M)*(2*hidden_dim)
        w_g_star = self.w_g_star(g_star)  # B*(49*M)*1
        w_g_star = w_g_star.squeeze(2) # B*(49*M)

        w_s_t = self.w_s_t(s_t_hat)  # B*(49*M)
        # F.tanh改为了 torch.tanh
        e_a = torch.tanh(w_g_star + w_s_t + coverage_img)  # B*(49*M)
        e_a = self.v(e_a)  # B*(49*M)
        attr_img = torch.softmax(e_a, dim=1)  # B*(49*M)

        attr_img = attr_img.unsqueeze(1).contiguous()  # B*1*(49*M)
        c_img = torch.bmm(attr_img, g_star)   # B*1*(2*hidden_dim)
        c_img = c_img.view(-1, config.hidden_dim * 2).contiguous()  # B*(2*hidden_dim)
        attr_img = attr_img.squeeze(1).contiguous()  # B*M
        coverage_img = coverage_img + attr_img

        return c_img, attr_img, coverage_img, ()

    def HAN_forward(self, global_features, local_features, s_t_hat, coverage_img, coverage_patches):
        """
         :param local_features: (tensor) B*M*512*49
         :param s_t_hat: B*(2*hidden_dim)
         :param coverage_img: B*M
         :param coverage_patches : B*M*512
         :return: c_img : B*(2*hidden_dim)
                  coverage_img : B*M
                  attr_img : B*M
                  attr_img_patches B*M*512
                  coverage_patches B*M*512
        """
        tensor_shape(s_t_hat)
        g_star = self.w_g(local_features)  # B*M*512*(2*hidden_dim)
        g_star = self.g_star(g_star)  # B*M*512*1
        g_star = g_star.squeeze(3).contiguous()  # B*M*512
        w_g_star = self.w_g_star(g_star)  # B*M*512
        w_s_t = self.w_s_t(s_t_hat)  # B*(M*512)
        w_s_t = w_s_t.view(-1, config.maxinum_imgs, 512)  # B*M*512
        # F.tanh改为了 torch.tanh
        e_a = torch.tanh(w_g_star + w_s_t + coverage_patches)  # B*M
        e_a = self.v(e_a)  # B*M*512
        attr_img_patches = torch.softmax(e_a, dim=1)  # B*M*512
        coverage_patches += attr_img_patches

        w_g_star_i = self.w_g_star_i(g_star)  # M*B
        w_g_star_i = w_g_star_i.squeeze(2)
        w_s_t_i = self.w_s_t_i(s_t_hat)  # B*M
        e_a_i = torch.tanh(w_g_star_i + w_s_t_i + coverage_img)
        e_a_i = self.v_i(e_a_i)  # B*M
        attr_img = torch.softmax(e_a_i, dim=1)  # B*M
        coverage_img += attr_img

        attr_img = attr_img.unsqueeze(1).contiguous()  # B*1*M

        c_img = torch.bmm(attr_img, g_star)  # B*1*(2*hidden_dim)
        c_img = c_img.view(-1, config.hidden_dim * 2).contiguous()  # B*(2*hidden_dim)
        attr_img = attr_img.squeeze(1).contiguous()  # B*M
        return c_img, attr_img, coverage_img, (attr_img_patches, coverage_patches)

    def forward(self, global_features, local_features, s_t_hat, coverage_img, coverage_patches):
        """
        :param global_features: (tensor) (B*M)*4096
        :param local_features: (tensor) (B*M)*512*49
        :param s_t_hat: B*(2*hidden_dim)
        :param coverage_img: B*M
        :param c_i: B*1*(2*hidden_dim)
        :param encoder_outputs:
        :return: c_img : B*1*(2*hidden_dim)
                coverage_img : B*M
                attr_img : B*M
        """
        model_forward_dict = {
            'ATG': self.ATG_forward,
            'ATL': self.ATL_forward,
            'HAN': self.HAN_forward
        }
        return model_forward_dict[self.img_attention_model](global_features, local_features, s_t_hat, coverage_img, coverage_patches)


class multi_attention(nn.Module):
    """
    mutil attention layer
    """
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

        e_txt = self.v_txt(w_c_txt + w_s_txt)  # B*1
        e_img = self.v_img(w_c_img + w_s_img)  # B*1
        # need dim arg
        alpha_txt = torch.softmax(e_txt, dim=1)  # B*1
        alpha_img = torch.softmax(e_img, dim=1)  # B*1

        c_mm = torch.mul(alpha_txt, input_c_txt) + torch.mul(alpha_img, input_c_img)  # B*(2*hidden_dim)
        return c_mm


class txt_reduce_state(nn.Module):
    def __init__(self):
        super(txt_reduce_state, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim

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
        # B*t_k*(2*hidden_dim)
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


class msmo_decoder(nn.Module):
    def __init__(self, img_attention_model=config.img_attention_model):
        super(msmo_decoder, self).__init__()
        self.img_attention_model = img_attention_model
        self.txt_attention = txt_attention()
        self.img_attention = img_attention(img_attention_model)
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
                c_mm, extra_zeros, enc_batch_extend_vocab, coverage_txt, step, golbal_features, local_features,
                coverage_img, coverage_img_patches):
        # 如果不是模型不是处于训练阶段
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.txt_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                       enc_padding_mask, coverage_txt)

            coverage_txt = coverage_next
            pass

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_mm, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_txt_next = self.txt_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage_txt)

        c_i, attn_img, coverage_img_next, img_patches_next = self.img_attention(golbal_features, local_features,
                                                                           s_t_hat, coverage_img, coverage_img_patches)


        c_mm = self.multi_attention(c_t, c_i, s_t_hat)

        # 模型处于训练阶段
        if self.training or step > 0:
            coverage_txt = coverage_txt_next
            coverage_img = coverage_img_next
            img_patches = img_patches_next

        p_gen = None
        if config.pointer_gen:
            # 多模态
            p_gen_input = torch.cat((c_mm, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_mm), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hid   den_dim

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

        return final_dist, s_t, c_mm, attn_dist, p_gen, coverage_txt, attn_img, coverage_img, img_patches


class MSMO(object):
    def __init__(self, model_file_path=None, is_eval=False, img_attention_model=config.img_attention_model):
        model_txt_encode = txt_encoder()
        model_reduce_state = txt_reduce_state()
        model_img_encode = img_encoder()
        model_decoder = msmo_decoder(img_attention_model=img_attention_model)

        # shared the embedding between encoder and decoder
        model_decoder.embedding.weight = model_txt_encode.embedding.weight
        # only forward calculate
        if is_eval:
            model_txt_encode = model_txt_encode.eval()
            model_reduce_state = model_reduce_state.eval()
            model_img_encode = model_img_encode.eval()
            model_decoder = model_decoder.eval()

        if use_cuda:
            model_txt_encode = model_txt_encode.cuda()
            model_img_encode = model_img_encode.cuda()
            model_decoder = model_decoder.cuda()
            model_reduce_state = model_reduce_state.cuda()

        self.txt_encode = model_txt_encode
        self.img_encode = model_img_encode
        self.decoder = model_decoder
        self.reduce_state = model_reduce_state
        self.img_attention_model = img_attention_model

        # load model state
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.txt_encode.load_state_dict(state['txt_encode_state_dict'])
            self.img_encode.load_state_dict(state['img_encode_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
        pass
