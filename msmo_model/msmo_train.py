# -*- coding:utf-8 -*-
# @Time： 2020-06-24 13:52
# @Author: Joshua_yi
# @FileName: msmo_train.py
# @Software: PyCharm
# @Project: MSMO

import os
import time
import torch
import torch.optim
from torch.nn.utils import clip_grad_norm_
import config
from msmo_model.msmo import MSMO
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from msmo_model.train_util import get_input_from_batch, get_output_from_batch, tensor_shape, print_info
import logging

logging.basicConfig(level=logging.INFO,  # logging level
                    filename=config.msmo_logging_path,
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

use_cuda = config.use_gpu and torch.cuda.is_available()


class Train(object):
    """
    train the model
    """
    def __init__(self, img_attention_model=config.img_attention_model):
        print_info(f'beign train the {img_attention_model} ...')

        self.img_attention_model = img_attention_model
        print_info('vocab generate ...')
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.vocab.write_metadata(fpath=config.word_id_path)
        self.pictures = []
        print_info('vocab generate finish')
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)

        pass

    def save_model(self, iter, model_filepath=config.msmo_modle_path):
        state = {
            'txt_encode_state_dict': self.model.txt_encode.state_dict(),
            'img_encode_state_dict': self.model.img_encode.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
        }
        model_save_path = os.path.join(model_filepath, 'model_%d_%d.pth' % (iter, int(time.time())))
        torch.save(state, model_save_path)
        pass

    def setup_train(self):
        self.model = MSMO(img_attention_model=self.img_attention_model)

        params = list(self.model.txt_encode.parameters()) + list(self.model.decoder.parameters()) + list(self.model.reduce_state.parameters())

        initial_lr = config.lr_coverage if config.is_coverage else config.lr

        self.optimizer = torch.optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0
        return start_iter, start_loss

    def train_one_batch(self, batch):
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
            final_dist, s_t_1, c_mm, attn_dist, p_gen, next_coverage_txt, attn_img, next_coverage_img, img_patches = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_mm,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage_txt, di, img_global_features, img_local_features, coverage_img, coverage_img_patches)



            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

            step_loss = -torch.log(gold_probs + config.eps) #B
            if config.is_coverage:
                step_coverage_loss_txt = torch.sum(torch.min(attn_dist, coverage_txt), 1) # B
                step_coverage_loss_img = torch.sum(torch.min(attn_img, coverage_img), 1) # B
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss_txt + config.cov_loss_wt * step_coverage_loss_img
                coverage_txt = next_coverage_txt
                coverage_img = next_coverage_img
                if config.img_attention_model == 'HAN':

                    step_coverage_loss_img_patches = torch.sum(torch.min(img_patches[0], img_patches[1]), (1, 2))
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss_img_patches
                    tensor_shape(step_loss)
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

        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.model.txt_encode.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item()

    def train_iters(self, iters=config.max_iterations):
        iter, running_avg_loss = self.setup_train()
        start = time.time()
        print_info('begin train ...')
        while iter < iters:

            iter += 1
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)
            # running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print_info(f'iter{iter} running_avg_loss is {running_avg_loss}')
            print_info(f'iter{iter} loss is {loss}')

            print_interval = 1
            if iter % print_interval == 0:
                print_info('steps %d, seconds for %d batch: %.2f s , loss: %f' % (iter, print_interval, time.time() - start, loss))
                start = time.time()
        print_info(f'end train {self.img_attention_model} model')
        pass


if __name__ == '__main__':
    train_processor = Train(img_attention_model=config.img_attention_model)
    train_processor.train_iters(iters=30)
    train_processor.save_model(30, model_filepath=config.modle_path)


