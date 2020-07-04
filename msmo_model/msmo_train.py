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
from data_util import config
from msmo_model.msmo import Model
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from msmo_model.train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()


class Train(object):
    """
    train the model
    """
    def __init__(self):
        print('vocab generate ...')
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        # self.vocab.write_metadata(fpath=config.word_id_path)

        print('vocab generate finish')
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        print('time sleep ...')
        time.sleep(1)
        print('time sleep end')
        train_dir = os.path.join(config.log_root, '/train_%d' % (int(time.time())))

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
    pass

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'txt_encode_state_dict': self.model.txt_encode.state_dict(),
            # FIXME (ly, 20200701): 可能不用保存img encode的参数
            'img_encode_state_dict': self.model.img_encode.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)
        # 训练模型


        params = list(self.model.txt_encode.parameters()) + list(self.model.decoder.parameters()) + list(self.model.reduce_state.parameters())

        initial_lr = config.lr_coverage if config.is_coverage else config.lr

        self.optimizer = torch.optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, imgs, c_i, coverage_img = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)



        encoder_outputs, encoder_feature, encoder_hidden = self.model.txt_encode(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        img_local_features, img_global_features = self.model.img_encode(imgs)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            # final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage,attn_img,next_coverage_img = self.model.decoder(y_t_1, s_t_1,
            #                                                                                encoder_outputs,
            #                                                                                encoder_feature,
            #                                                                                enc_padding_mask, c_t_1,
            #                                                                                extra_zeros,
            #                                                                                enc_batch_extend_vocab,
            #                                                                                coverage, di, img_global_features, img_local_features, coverage_img,c_i)

            final_dist, attn_dist, next_coverage, attn_img, next_coverage_img = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di, img_global_features, img_local_features, coverage_img,c_i)

            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss_txt = torch.sum(torch.min(attn_dist, coverage), 1)
                step_coverage_loss_img = torch.sum(torch.min(attn_img,coverage_img),1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss_txt + config.cov_loss_wt * step_coverage_loss_img
                coverage = next_coverage
                coverage_img = next_coverage_img
                # TODO (20200703): 增加HAN模型的loss
                if config.img_attention_model == 'HAN':
                    pass

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.model.txt_encode.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            iter += 1
            print('%i iters...' % iter)
            batch = self.batcher.next_batch()
            print('train one batch...')
            loss = self.train_one_batch(batch)
            print('calc_running_avg_loss...')
            # running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)

            # 每迭代100log更新一次
            # if iter % 100 == 0:
            #     self.summary_writer.flush()
            # 间隔多少次打印一次

            print_interval = 1
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f s , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()

            if iter % 50 == 0:
                self.save_model(running_avg_loss, iter)


if __name__ == '__main__':
    train_processor = Train()
    train_processor.trainIters(config.max_iterations)
