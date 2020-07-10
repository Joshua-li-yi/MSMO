# -*- coding:utf-8 -*-
# @Time： 2020-07-09 22:10
# @Author: Joshua_yi
# @FileName: eval.py
# @Software: PyCharm
# @Project: MSMO
# @Description: 

import os
import time

import tensorflow as tf
import torch

import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from msmo_model.train_util import get_input_from_batch, get_output_from_batch
from msmo_model.msmo import MSMO

use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(1)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        # self.summary_writer = tf.summary.FileWriter(eval_dir)

        self.model = MSMO(model_file_path=model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_mm, coverage_txt, imgs, coverage_img, coverage_img_patches = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.txt_encode(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        img_local_features, img_global_features = self.model.img_encode(imgs)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_mm, attn_dist, p_gen, next_coverage_txt, attn_img, next_coverage_img, img_patches = self.model.decoder(
                y_t_1, s_t_1,
                encoder_outputs,
                encoder_feature,
                enc_padding_mask, c_mm,
                extra_zeros,
                enc_batch_extend_vocab,
                coverage_txt, di, img_global_features, img_local_features, coverage_img, coverage_img_patches)

            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

            step_loss = -torch.log(gold_probs + config.eps)  # B
            if config.is_coverage:
                step_coverage_loss_txt = torch.sum(torch.min(attn_dist, coverage_txt), 1)  # B
                step_coverage_loss_img = torch.sum(torch.min(attn_img, coverage_img), 1)  # B
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss_txt + config.cov_loss_wt * step_coverage_loss_img
                coverage_txt = next_coverage_txt
                coverage_img = next_coverage_img
                if config.img_attention_model == 'HAN':
                    step_coverage_loss_img_patches = torch.sum(torch.min(img_patches[0], img_patches[1]), (1, 2))
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss_img_patches
                    coverage_img_patches = img_patches[1]
                    pass

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
            pass

        self.pictures.append(torch.argmax(coverage_img))

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.data[0]

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                    iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()