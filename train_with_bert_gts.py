# -*- coding: utf-8 -*-
from optimization import BERTAdam
import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
import pdb
from bert_bucket_iterator import BucketIterator
from sklearn import metrics
from bert_data_utils_gts import ABSADatasetReader
from bert_models import BERT_GCN, BERT_GCN0, BERT_GCN3, BERT_GCN03
# from evaluation3 import get_metric, find_pair, find_term,  compute_sentiment, find_pair_sentiment, find_grid_term, find_pair_according_to_ao, find_triple_sentiment_according_to_ao, find_triple_sentiment_according_to_pair, find_pair_according_to_all_span, find_pair_according_to_ao_gts, find_triple_sentiment_according_to_pair_gts, find_triple_sentiment_according_to_ao_gts
from evaluation3 import find_term_span, find_pair_according_to_ao_span, find_triple_sentiment_according_to_ao_span, find_triple_sentiment_according_to_pair_span
from utils import *
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        import datetime as dt
        now_time = dt.datetime.now().strftime('%F %T')
        self.absa_dataset = ABSADatasetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        # adj, features = load_corpus(dataset_str=opt.dataset)
        self.train_data_loader = BucketIterator(data=self.absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = BucketIterator(data=self.absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = BucketIterator(data=self.absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False, sort=False)
        
        self.model = opt.model_class(opt, 'bert-base-uncased').to(opt.device)#opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self.f_out = open('log_bert/'+self.opt.model_name+'_'+self.opt.dataset+'_val'+str(now_time)+'.txt', 'w', encoding='utf-8')
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # def _reset_params(self):
    #     for p in self.model.parameters():
    #         if p.requires_grad:
    #             if len(p.shape) > 1:
    #                 self.opt.initializer(p)
    #             else:
    #                 stdv = 1. / math.sqrt(p.shape[0])
    #                 torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    def _reset_params(self):
        for p in self.model.aspect_opinion_classifier.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        
        for p in self.model.triple_classifier.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_aspect_dev_f1, max_opinion_dev_f1 = 0, 0
        max_pair_dev_f1, max_pair_sentiment_dev_f1 = 0, 0
        max_pair_sentiment_dev_f1_macro = 0 

        max_aspect_test_f1, max_opinion_test_f1 = 0, 0
        max_pair_test_f1, max_pair_sentiment_test_f1 = 0, 0
        max_pair_sentiment_test_f1_macro = 0

        global_step = 0
        continue_not_increase = 0
        best_results, best_labels = [], []
        for epoch in (range(self.opt.num_epoch)):
            print('>' * 100)
            print('epoch: ', epoch)
            self.f_out.write('>' * 100+'\n')
            self.f_out.write('epoch: {:.4f}\n'.format(epoch))
            loss_g_a, loss_g_o, loss_g_s, loss_g_ag, loss_g_og, loss_g_p, loss_g_ps = 0, 0, 0, 0, 0, 0, 0
            correct_g, predicted_g, relevant_g = 0, 0, 0
            
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                targets_aspect_sequence = sample_batched['aspect_span_labels'].to(self.opt.device)
                targets_opinion_sequence = sample_batched['opinion_span_labels'].to(self.opt.device)
                targets_pair = sample_batched['pair_grid_labels'].to(self.opt.device)
                targets_pair_sentiment = sample_batched['triple_grid_labels'].to(self.opt.device)
                
                mask = sample_batched['mask'].to(self.opt.device)
                # aspect_mask = sample_batched['aspect_mask'].to(self.opt.device)
                # aspect_mask_ = aspect_mask.reshape(-1).long()
                
                bs, sen_len = mask.size()
                mask_grid = torch.where(mask.unsqueeze(1).repeat(1,sen_len,1) + mask.unsqueeze(-1).repeat(1,1,sen_len) == 2,\
                                        torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                        torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                outputs_aspect, outputs_opinion, outputs_pair, outputs_pair_sentiment = self.model(inputs, mask)

                outputs_aspect_env = outputs_aspect.argmax(dim=-1)
                outputs_aspect_env = outputs_aspect_env.view(targets_pair.shape[0],targets_pair.shape[1],targets_pair.shape[2])
                outputs_aspect_, targets_aspect_ = outputs_aspect.reshape(-1,3), targets_aspect_sequence.reshape(-1).long()

                outputs_opinion_env = outputs_opinion.argmax(dim=-1)
                outputs_opinion_env = outputs_opinion_env.view(targets_pair.shape[0],targets_pair.shape[1],targets_pair.shape[2])
                outputs_opinion_, targets_opinion_ = outputs_opinion.reshape(-1,3), targets_opinion_sequence.reshape(-1).long()

                outputs_pair_env = outputs_pair.argmax(dim=-1)
                outputs_pair_env = outputs_pair_env.view(targets_pair.shape[0],targets_pair.shape[1],targets_pair.shape[2])
                outputs_pair_, targets_pair_ = outputs_pair.reshape(-1,3), targets_pair.reshape(-1).long()

                outputs_pair_sentiment_env = outputs_pair_sentiment.argmax(dim=-1)
                outputs_pair_sentiment_env = outputs_pair_sentiment_env.view(targets_pair_sentiment.shape[0],targets_pair_sentiment.shape[1],targets_pair_sentiment.shape[2])
                outputs_pair_sentiment_, targets_pair_sentiment_ = outputs_pair_sentiment.reshape(-1,4), targets_pair_sentiment.reshape(-1).long()
                
                loss_aspect = F.cross_entropy(outputs_aspect_, targets_aspect_, ignore_index=-1)
                loss_opinion = F.cross_entropy(outputs_opinion_, targets_opinion_, ignore_index=-1)
                loss_pair = F.cross_entropy(outputs_pair_, targets_pair_, ignore_index=-1)
                loss_pair_sentiment = F.cross_entropy(outputs_pair_sentiment_, targets_pair_sentiment_, ignore_index=-1)
                # loss = loss_aspect + loss_opinion + loss_pair + loss_pair_sentiment
                # loss = loss_aspect + loss_opinion
                # loss = loss_pair_sentiment
                loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                    loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 

                if self.opt.freeze == 1:
                    if epoch > 10 and epoch < 20:
                        loss = 0*loss_aspect + 0*loss_opinion + loss_pair  + loss_pair_sentiment 
                    elif epoch <=10:
                        loss = loss_aspect + loss_opinion + 0*loss_pair  + 0*loss_pair_sentiment
                    else:
                        loss = loss_aspect + loss_opinion + loss_pair  + loss_pair_sentiment
                else:
                    loss = loss_aspect + loss_opinion + loss_pair  + loss_pair_sentiment
                loss.backward()
                optimizer.step()

            dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro, dev_loss = self._evaluate_acc_f1()
            test_f_aspect, test_f_opinion, test_f_pair, [test_f_pair_sentiment, test_p_pair_sentiment, test_r_pair_sentiment], test_f_pair_sentiment_macro, results, labels, test_loss = self._test_acc_f1()

            print('train loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(loss_g_a.item(), loss_g_o.item(), loss_g_p.item(), loss_g_ps.item()))
            print('dev loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(dev_loss[0].item(), dev_loss[1].item(), dev_loss[2].item(), dev_loss[3].item()))
            print('dev: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}'.format(dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro))
            print('test loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(test_loss[0].item(), test_loss[1].item(), test_loss[2].item(), test_loss[3].item()))
            print('test: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}'.format(test_f_aspect, test_f_opinion, test_f_pair, test_f_pair_sentiment, test_f_pair_sentiment_macro))

            self.f_out.write('train loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(loss_g_a.item(), loss_g_o.item(), loss_g_p.item(), loss_g_ps.item()))
            self.f_out.write('dev loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(dev_loss[0].item(), dev_loss[1].item(), dev_loss[2].item(), dev_loss[3].item()))
            self.f_out.write('dev: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}\n'.format(dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro))
            self.f_out.write('test loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(test_loss[0].item(), test_loss[1].item(), test_loss[2].item(), test_loss[3].item()))
            self.f_out.write('test: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}\n'.format(test_f_aspect, test_f_opinion, test_f_pair, test_f_pair_sentiment, test_f_pair_sentiment_macro))
            
            self.f_out.write('test: p-pair-sentiment: {:.4f}, r-pair-sentiment: {:.4f}\n'\
                .format(test_p_pair_sentiment, test_r_pair_sentiment))
            if dev_f_pair_sentiment > max_pair_sentiment_dev_f1:
                max_pair_dev_f1 = dev_f_pair
                max_aspect_dev_f1 = dev_f_aspect
                max_opinion_dev_f1 = dev_f_opinion
                max_pair_sentiment_dev_f1 = dev_f_pair_sentiment
                max_pair_sentiment_dev_f1_macro = dev_f_pair_sentiment_macro
                best_model = self.model

                max_pair_test_f1 = test_f_pair
                max_aspect_test_f1 = test_f_aspect
                max_opinion_test_f1 = test_f_opinion
                max_pair_sentiment_test_f1 = test_f_pair_sentiment
                max_pair_sentiment_test_f1_macro = test_f_pair_sentiment_macro
                best_results = results
                best_labels = labels
                self.f_out.write('dev: {:.4f}, test: {:.4f}'.format(max_pair_sentiment_dev_f1, max_pair_sentiment_test_f1))
        return max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1,\
                max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1,\
                 best_results, best_labels, best_model

    def _evaluate_acc_f1(self):
            # switch model to evaluation mode
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            predicted_p, relevant_p, correct_p = 0, 0, 0
            predicted_ps, relevant_ps, correct_ps = 0, 0, 0
            predicted_ps_, relevant_ps_, correct_ps_ = 0, 0, 0
            predicted_a, relevant_a, correct_a = 0, 0, 0
            predicted_o, relevant_o, correct_o = 0, 0, 0

            predicted_ps_macro, relevant_ps_macro, correct_ps_macro = {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}
            dic = {1:'pos', 2:'neg', 3:'neu'}

            loss_g_a, loss_g_o, loss_g_s, loss_g_p, loss_g_ps, loss_g_ag, loss_g_og = 0, 0, 0, 0, 0, 0, 0
            with torch.no_grad():
                for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                    t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                    t_targets_aspect = t_sample_batched['aspect_span_labels'].to(self.opt.device)
                    t_targets_opinion = t_sample_batched['opinion_span_labels'].to(self.opt.device)
                    t_targets_sentiment = t_sample_batched['sentiment_sequence_labels'].to(self.opt.device)
                    t_targets_pair = t_sample_batched['pair_grid_labels'].to(self.opt.device)
                    t_targets_pair_sentiment = t_sample_batched['triple_grid_labels'].to(self.opt.device)
                    t_targets_mask = t_sample_batched['mask'].to(self.opt.device)
                    
                    bs, sen_len = t_targets_mask.size()
                    t_targets_mask_grid = torch.where(t_targets_mask.unsqueeze(1).repeat(1,sen_len,1) + t_targets_mask.unsqueeze(-1).repeat(1,1,sen_len) == 2,\
                                            torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                            torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                    t_outputs_aspect, t_outputs_opinion, t_outputs_pair, t_outputs_pair_sentiment = self.model(t_inputs, t_targets_mask)
                    
                    t_outputs_aspect_env = t_outputs_aspect.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                    t_outputs_opinion_env = t_outputs_opinion.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                    t_outputs_pair_env = t_outputs_pair.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                    t_outputs_pair_sentiment_env = t_outputs_pair_sentiment.argmax(dim=-1).view(t_targets_pair_sentiment.shape[0],t_targets_pair_sentiment.shape[1],t_targets_pair_sentiment.shape[2])
                    # compute loss 
                    outputs_aspect_, targets_aspect_ = t_outputs_aspect.reshape(-1,3), t_targets_aspect.reshape(-1).long()
                    outputs_opinion_, targets_opinion_ = t_outputs_opinion.reshape(-1,3), t_targets_opinion.reshape(-1).long()
                    outputs_pair_, targets_pair_ = t_outputs_pair.reshape(-1,3), t_targets_pair.reshape(-1).long()
                    outputs_pair_sentiment_, targets_pair_sentiment_ = t_outputs_pair_sentiment.reshape(-1,4), t_targets_pair_sentiment.reshape(-1).long()

                    loss_aspect = F.cross_entropy(outputs_aspect_, targets_aspect_, ignore_index=-1)
                    loss_opinion = F.cross_entropy(outputs_opinion_, targets_opinion_, ignore_index=-1)
                    loss_pair = F.cross_entropy(outputs_pair_, targets_pair_, ignore_index=-1)
                    loss_pair_sentiment = F.cross_entropy(outputs_pair_sentiment_, targets_pair_sentiment_, ignore_index=-1)
                    
                    loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                        loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 
                    # metrics
                    outputs_a = (t_outputs_aspect_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_a = t_targets_aspect.cpu().numpy().tolist()

                    outputs_o = (t_outputs_opinion_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_o = t_targets_opinion.cpu().numpy().tolist()
                    
                    outputs_p = (t_outputs_pair_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_p = t_targets_pair.cpu().numpy().tolist()

                    outputs_ps = (t_outputs_pair_sentiment_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_ps = t_targets_pair_sentiment.cpu().numpy().tolist()

                    for out_a, tar_a, out_o, tar_o, out_p, tar_p, out_ps, tar_ps in zip(outputs_a, targets_a, outputs_o, targets_o,  outputs_p, targets_p, outputs_ps, targets_ps):   
                        
                        predict_aspect = find_term_span(out_a)
                        true_aspect = find_term_span(tar_a)

                        predict_opinion = find_term_span(out_o)
                        true_opinion = find_term_span(tar_o)

                        predict_pairs = find_pair_according_to_ao_span(out_p, predict_aspect, predict_opinion)
                        true_pairs = find_pair_according_to_ao_span(tar_p, true_aspect, true_opinion) 

                        predict_triple_1 = find_triple_sentiment_according_to_ao_span(out_ps, predict_aspect, predict_opinion)
                        true_triple_1 = find_triple_sentiment_according_to_ao_span(tar_ps, true_aspect, true_opinion) 

                        predict_triple_2 = find_triple_sentiment_according_to_pair_span(out_ps, predict_pairs)
                        true_triple_2 = find_triple_sentiment_according_to_pair_span(tar_ps, true_pairs) 

                        predicted_a += len(predict_aspect)
                        relevant_a += len(true_aspect)
                        for aspect in predict_aspect:
                            if aspect in true_aspect:
                                correct_a += 1

                        predicted_o += len(predict_opinion)
                        relevant_o += len(true_opinion)
                        for opinion in predict_opinion:
                            if opinion in true_opinion:
                                correct_o += 1

                        predicted_p += len(predict_pairs)
                        relevant_p += len(true_pairs)
                        for pair in predict_pairs:
                            if pair in true_pairs:
                                correct_p += 1  

                        predicted_ps_ += len(predict_triple_1)
                        relevant_ps_ += len(true_triple_1)
                        for triple in predict_triple_1:
                            if triple in true_triple_1:
                                correct_ps_ += 1

                        predicted_ps += len(predict_triple_2)
                        relevant_ps += len(true_triple_2)
                        for triple in predict_triple_2:
                            if triple in true_triple_2:
                                correct_ps += 1            
                # micro
                p_pair_sentiment = correct_ps / (predicted_ps + 1e-6)
                r_pair_sentiment = correct_ps / (relevant_ps + 1e-6)
                f_pair_sentiment = 2 * p_pair_sentiment * r_pair_sentiment / (p_pair_sentiment + r_pair_sentiment + 1e-6)
                # macro
                p_pair_sentiment_ = correct_ps_ / (predicted_ps_ + 1e-6)
                r_pair_sentiment_ = correct_ps_ / (relevant_ps_ + 1e-6)
                f_pair_sentiment_ = 2 * p_pair_sentiment_ * r_pair_sentiment_ / (p_pair_sentiment_ + r_pair_sentiment_ + 1e-6)
                f_pair_sentiment_macro = f_pair_sentiment_

                p_pair = correct_p / (predicted_p + 1e-6)
                r_pair = correct_p / (relevant_p + 1e-6)
                f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-6)

                p_aspect = correct_a / (predicted_a + 1e-6)
                r_aspect = correct_a / (relevant_a + 1e-6)
                f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

                p_opinion = correct_o / (predicted_o + 1e-6)
                r_opinion = correct_o / (relevant_o + 1e-6)
                f_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-6)

                return f_aspect, f_opinion, f_pair, f_pair_sentiment, f_pair_sentiment_macro, [loss_g_a, loss_g_o, loss_g_p, loss_g_ps]

    def _test_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        predicted_p, relevant_p, correct_p = 0, 0, 0
        predicted_ps, relevant_ps, correct_ps = 0, 0, 0
        predicted_ps_, relevant_ps_, correct_ps_ = 0, 0, 0
        predicted_a, relevant_a, correct_a = 0, 0, 0
        predicted_o, relevant_o, correct_o = 0, 0, 0

        predicted_ps_macro, relevant_ps_macro, correct_ps_macro = {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}
        dic = {1:'pos', 2:'neg', 3:'neu'}

        loss_g_a, loss_g_o, loss_g_p, loss_g_ps = 0, 0, 0, 0

        aspect_results, opinion_results, sentiment_results, pair_results, pair_sentiment_results = [], [], [], [], [] 
        aspect_labels, opinion_labels, sentiment_labels, pair_labels, pair_sentiment_labels = [], [], [], [], [] 

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets_pair = t_sample_batched['pair_grid_labels'].to(self.opt.device)
                t_targets_pair_sentiment = t_sample_batched['triple_grid_labels'].to(self.opt.device)
                t_targets_aspect = t_sample_batched['aspect_span_labels'].to(self.opt.device)
                t_targets_opinion = t_sample_batched['opinion_span_labels'].to(self.opt.device)
                t_targets_mask = t_sample_batched['mask'].to(self.opt.device)

                bs, sen_len = t_targets_mask.size()
                t_targets_mask_grid = torch.where(t_targets_mask.unsqueeze(1).repeat(1,sen_len,1) + t_targets_mask.unsqueeze(-1).repeat(1,1,sen_len) == 2,\
                                        torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                        torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                t_outputs_aspect, t_outputs_opinion, t_outputs_pair, t_outputs_pair_sentiment = self.model(t_inputs, t_targets_mask)
                
                t_outputs_aspect_env = t_outputs_aspect.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                t_outputs_opinion_env = t_outputs_opinion.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                t_outputs_pair_env = t_outputs_pair.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                t_outputs_pair_sentiment_env = t_outputs_pair_sentiment.argmax(dim=-1).view(t_targets_pair_sentiment.shape[0],t_targets_pair_sentiment.shape[1],t_targets_pair_sentiment.shape[2])
                # compute loss 
                outputs_aspect_, targets_aspect_ = t_outputs_aspect.reshape(-1,3), t_targets_aspect.reshape(-1).long()
                outputs_opinion_, targets_opinion_ = t_outputs_opinion.reshape(-1,3), t_targets_opinion.reshape(-1).long()
                outputs_pair_, targets_pair_ = t_outputs_pair.reshape(-1,3), t_targets_pair.reshape(-1).long()
                outputs_pair_sentiment_, targets_pair_sentiment_ = t_outputs_pair_sentiment.reshape(-1,4), t_targets_pair_sentiment.reshape(-1).long()

                loss_aspect = F.cross_entropy(outputs_aspect_, targets_aspect_, ignore_index=-1)
                loss_opinion = F.cross_entropy(outputs_opinion_, targets_opinion_, ignore_index=-1)
                loss_pair = F.cross_entropy(outputs_pair_, targets_pair_, ignore_index=-1)
                loss_pair_sentiment = F.cross_entropy(outputs_pair_sentiment_, targets_pair_sentiment_, ignore_index=-1)

                loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                    loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 
                # metrics
                outputs_a = (t_outputs_aspect_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_a = t_targets_aspect.cpu().numpy().tolist()

                outputs_o = (t_outputs_opinion_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_o = t_targets_opinion.cpu().numpy().tolist()
                
                outputs_p = (t_outputs_pair_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_p = t_targets_pair.cpu().numpy().tolist()

                outputs_ps = (t_outputs_pair_sentiment_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_ps = t_targets_pair_sentiment.cpu().numpy().tolist()

                for out_a, tar_a, out_o, tar_o, out_p, tar_p, out_ps, tar_ps in zip(outputs_a, targets_a, outputs_o, targets_o,  outputs_p, targets_p, outputs_ps, targets_ps):   
                        
                    predict_aspect = find_term_span(out_a)
                    true_aspect = find_term_span(tar_a)

                    predict_opinion = find_term_span(out_o)
                    true_opinion = find_term_span(tar_o)

                    predict_pairs = find_pair_according_to_ao_span(out_p, predict_aspect, predict_opinion)
                    true_pairs = find_pair_according_to_ao_span(tar_p, true_aspect, true_opinion) 

                    predict_triple_1 = find_triple_sentiment_according_to_ao_span(out_ps, predict_aspect, predict_opinion)
                    true_triple_1 = find_triple_sentiment_according_to_ao_span(tar_ps, true_aspect, true_opinion) 

                    predict_triple_2 = find_triple_sentiment_according_to_pair_span(out_ps, predict_pairs)
                    true_triple_2 = find_triple_sentiment_according_to_pair_span(tar_ps, true_pairs) 

                    predicted_a += len(predict_aspect)
                    relevant_a += len(true_aspect)
                    for aspect in predict_aspect:
                        if aspect in true_aspect:
                            correct_a += 1

                    predicted_o += len(predict_opinion)
                    relevant_o += len(true_opinion)
                    for opinion in predict_opinion:
                        if opinion in true_opinion:
                            correct_o += 1

                    predicted_p += len(predict_pairs)
                    relevant_p += len(true_pairs)
                    for pair in predict_pairs:
                        if pair in true_pairs:
                            correct_p += 1  

                    predicted_ps_ += len(predict_triple_1)
                    relevant_ps_ += len(true_triple_1)
                    for triple in predict_triple_1:
                        if triple in true_triple_1:
                            correct_ps_ += 1

                    predicted_ps += len(predict_triple_2)
                    relevant_ps += len(true_triple_2)
                    for triple in predict_triple_2:
                        if triple in true_triple_2:
                            correct_ps += 1

                # save results and labels
                aspect_results.append(t_outputs_aspect.view(t_targets_pair.shape[0], -1, 3).cpu().numpy().tolist())
                opinion_results.append(t_outputs_opinion.view(t_targets_pair.shape[0], -1, 3).cpu().numpy().tolist())
                pair_results.append(t_outputs_pair.view(t_targets_pair.shape[0], t_targets_aspect.shape[-1], t_targets_aspect.shape[-1], 3).cpu().numpy().tolist())
                pair_sentiment_results.append(t_outputs_pair_sentiment.view(t_targets_pair.shape[0], t_targets_aspect.shape[-1], t_targets_aspect.shape[-1], 4).cpu().numpy().tolist())

                aspect_labels.append(t_targets_aspect.cpu().numpy().tolist())
                opinion_labels.append(t_targets_opinion.cpu().numpy().tolist())
                pair_labels.append(t_targets_pair.cpu().numpy().tolist())
                pair_sentiment_labels.append(t_targets_pair_sentiment.cpu().numpy().tolist())

            print('triple:')
            print('predicted: {:.0f}, relevant: {:.0f}, correct: {:.0f}'.format(predicted_ps, relevant_ps, correct_ps))
            # p_pair_sentiment = correct_ps / (predicted_ps + 1e-6)
            # r_pair_sentiment = correct_ps / (relevant_ps + 1e-6)
            # f_pair_sentiment = 2 * p_pair_sentiment * r_pair_sentiment / (p_pair_sentiment + r_pair_sentiment + 1e-6)
            print('triple:')
            print('predicted: {:.0f}, relevant: {:.0f}, correct: {:.0f}'.format(predicted_ps_, relevant_ps_, correct_ps_))
            # p_pair_sentiment_ = correct_ps_ / (predicted_ps_ + 1e-6)
            # r_pair_sentiment_ = correct_ps_ / (relevant_ps_ + 1e-6)
            # f_pair_sentiment_ = 2 * p_pair_sentiment_ * r_pair_sentiment_ / (p_pair_sentiment_ + r_pair_sentiment_ + 1e-6)
            # f_pair_sentiment_macro = f_pair_sentiment_
            print('pair:')
            print('predicted: {:.0f}, relevant: {:.0f}, correct: {:.0f}'.format(predicted_p, relevant_p, correct_p))
            # p_pair = correct_p / (predicted_p + 1e-6)
            # r_pair = correct_p / (relevant_p + 1e-6)
            # f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-6)
            print('aspect:')
            print('predicted: {:.0f}, relevant: {:.0f}, correct: {:.0f}'.format(predicted_a, relevant_a, correct_a))
            # p_aspect = correct_a / (predicted_a + 1e-6)
            # r_aspect = correct_a / (relevant_a + 1e-6)
            # f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)
            print('opinion:')
            print('predicted: {:.0f}, relevant: {:.0f}, correct: {:.0f}'.format(predicted_o, relevant_o, correct_o))
            # p_opinion = correct_o / (predicted_o + 1e-6)
            # r_opinion = correct_o / (relevant_o + 1e-6)
            # f_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-6)
            # micro
            p_pair_sentiment = correct_ps / (predicted_ps + 1e-6)
            r_pair_sentiment = correct_ps / (relevant_ps + 1e-6)
            f_pair_sentiment = 2 * p_pair_sentiment * r_pair_sentiment / (p_pair_sentiment + r_pair_sentiment + 1e-6)
            # macro
            p_pair_sentiment_ = correct_ps_ / (predicted_ps_ + 1e-6)
            r_pair_sentiment_ = correct_ps_ / (relevant_ps_ + 1e-6)
            f_pair_sentiment_ = 2 * p_pair_sentiment_ * r_pair_sentiment_ / (p_pair_sentiment_ + r_pair_sentiment_ + 1e-6)
            f_pair_sentiment_macro = f_pair_sentiment_

            p_pair = correct_p / (predicted_p + 1e-6)
            r_pair = correct_p / (relevant_p + 1e-6)
            f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-6)

            p_aspect = correct_a / (predicted_a + 1e-6)
            r_aspect = correct_a / (relevant_a + 1e-6)
            f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

            p_opinion = correct_o / (predicted_o + 1e-6)
            r_opinion = correct_o / (relevant_o + 1e-6)
            f_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-6)

            results = [aspect_results, opinion_results, sentiment_results, pair_results, pair_sentiment_results]
            labels = [aspect_labels, opinion_labels, sentiment_labels, pair_labels, pair_sentiment_labels]

            return f_aspect, f_opinion, f_pair, [f_pair_sentiment, p_pair_sentiment, r_pair_sentiment], f_pair_sentiment_macro, results, labels, [loss_g_a, loss_g_o, loss_g_p, loss_g_ps]

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        if not os.path.exists('log/'):
            os.mkdir('log/')

        import datetime as dt
        now_time = dt.datetime.now().strftime('%F %T')

        # f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val'+str(now_time)+'.txt', 'w', encoding='utf-8')
        
        # print args
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.f_out.write('n_trainable_params: {0}, n_nontrainable_params: {1}\n'.format(n_trainable_params, n_nontrainable_params)+'\n')
        self.f_out.write('> training arguments:\n')
        
        for arg in vars(self.opt):
            self.f_out.write('>>> {0}: {1}'.format(arg, getattr(self.opt, arg))+'\n')
        max_aspect_test_f1_avg = 0
        max_opinion_test_f1_avg = 0
        max_sentiment_test_f1_avg = 0
        max_absa_test_f1_avg = 0
        max_pair_test_f1_avg = 0
        max_pair_sentiment_test_f1_avg = 0
        for i in range(self.opt.repeats):
            repeats = self.opt.repeats
            print('repeat: ', (i+1))
            self.f_out.write('repeat: '+str(i+1)+'\n')
            # _params = filter(lambda p: p.requires_grad, self.model.parameters())
            # self._reset_params()
            _params = self.model.parameters()
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            # num_train_steps = int(len(self.absa_dataset.train_data) / opt.batch_size * self.opt.num_epoch) 
            # print(num_train_steps)
            # no_decay = ['bias', 'gamma', 'beta']
            # optimizer_parameters = [
            #         {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            #         {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            #         ]
            # optimizer = BERTAdam(optimizer_parameters, lr=opt.learning_rate, warmup=opt.warmup_proportion, t_total=num_train_steps)
            
            # optimizer = torch.optim.Adam([
            #     {'params': self.model.bert_model.parameters(), 'lr': 5e-5},
            #     {'params': self.model.aspect_opinion_classifier.parameters()},
            #     {'params': self.model.triple_classifier.parameters()}
            # ], lr=5e-5)
            # optimizer = torch.optim.Adam([
            #     {'params': self.model.bert_model.parameters(), 'lr': 1e-5},
            #     {'params': self.model.aspect_classifier.parameters()},
            #     {'params': self.model.opinion_classifier.parameters()},
            #     {'params': self.model.pair_classifier.parameters()},
            #     {'params': self.model.pair_sentiment_classifier.parameters()},
            # ], lr=1e-5)
            
            max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1,\
            max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1, best_results, best_labels, best_model = self._train(criterion, optimizer)

            if self.opt.save_model == 1:
                torch.save(best_model.bert_model, './save_bert_model/' + self.opt.model_name + '_' + str(max_pair_sentiment_test_f1) + '_' + self.opt.dataset + '.pkl')

            if self.opt.write_results == 1:
                results_a, results_o, results_s, results_p, results_ps = best_results
                labels_a, labels_o, labels_s, labels_p, labels_ps = best_labels
                # write results
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_a.npy', results_a)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_o.npy', results_o)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_s.npy', results_s)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_p.npy', results_p)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_ps.npy', results_ps)
                # write labels
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_a.npy', labels_a)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_o.npy', labels_o)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_s.npy', labels_s)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_p.npy', labels_p)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_ps.npy', labels_ps)
            print('max_aspect_dev_f1: {:.4f}, max_opinion_dev_f1: {:.4f}, max_pair_dev_f1: {:.4f}, max_pair_sentiment_dev_f1: {:.4f}'.format(max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1))
            print('max_aspect_test_f1: {:.4f}, max_opinion_test_f1: {:.4f}, max_pair_test_f1: {:.4f}, max_pair_sentiment_test_f1: {:.4f}'.format(max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1))
            self.f_out.write('max_aspect_dev_f1: {:.4f}, max_opinion_dev_f1: {:.4f}, max_pair_dev_f1: {:.4f}, max_pair_sentiment_dev_f1: {:.4f}\n'\
                .format(max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1)+'\n')
            self.f_out.write('max_aspect_test_f1: {:.4f}, max_opinion_test_f1: {:.4f}, max_pair_test_f1: {:.4f}, max_pair_sentiment_test_f1: {:.4f}\n'\
                .format(max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1)+'\n')
            max_aspect_test_f1_avg += max_aspect_test_f1
            max_opinion_test_f1_avg += max_opinion_test_f1
            max_pair_test_f1_avg += max_pair_test_f1
            max_pair_sentiment_test_f1_avg += max_pair_sentiment_test_f1
            print('#' * 100)

        print("max_aspect_test_f1_avg:", max_aspect_test_f1_avg / repeats)
        print("max_opinion_test_f1_avg:", max_opinion_test_f1_avg / repeats)
        print("max_pair_test_f1_avg:", max_pair_test_f1_avg / repeats)
        print("max_pair_sentiment_test_f1_avg:", max_pair_sentiment_test_f1_avg / repeats)

        self.f_out.write("max_aspect_test_f1_avg:"+ str(max_aspect_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_opinion_test_f1_avg:"+ str(max_opinion_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_pair_test_f1_avg:" + str(max_pair_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_pair_sentiment_test_f1_avg:" + str(max_pair_sentiment_test_f1_avg / repeats) + '\n')

        self.f_out.close()

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str)
    parser.add_argument('--dataset', default='res14', type=str, help='res14, lap14, res15')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--use_graph0', default=1, type=int)
    parser.add_argument('--use_graph1', default=0, type=int)
    parser.add_argument('--use_graph2', default=0, type=int)
    parser.add_argument('--use_graph3', default=0, type=int)
    parser.add_argument('--write_results', default=0, type=int)
    parser.add_argument('--emb_for_ao', default='private_single', type=str, help='private_single, private_multi, shared_multi' )
    parser.add_argument('--emb_for_ps', default='private_single', type=str, help='private_single, private_multi, shared_multi' )
    parser.add_argument('--use_aspect_opinion_sequence_mask', default=0, type=int, help='1: use the predicted aspect_sequence_label and opinion_sequence_label to construct a grid mask for pair prediction.' )
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training."')
    parser.add_argument('--gcn_layers_in_graph0', default=1, type=int, help='1 or 2' )
    parser.add_argument('--save_model', default=1, type=int)
    parser.add_argument('--if_lambda', default=0, type=int)
    parser.add_argument('--freeze', default=0, type=int)
    opt = parser.parse_args()

    model_classes = {
        'bert': BERT_GCN,
        'bert0': BERT_GCN0,
        'bert3': BERT_GCN3,
        'bert03': BERT_GCN03
    }
    input_colses = {
        'bert': ['text_indices', 'mask'],\
        'bert0': ['text_indices', 'mask'],\
        'bert3': ['text_indices', 'mask'],\
        'bert03': ['text_indices', 'mask'],\
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()