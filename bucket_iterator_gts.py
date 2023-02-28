# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
import numpy as np
import pdb

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices, batch_mask, batch_aspect_mask = [], [], []
        # batch_mask, batch_aspect_mask = [], []
        # batch_target_triple = []
        # batch_local_graph, batch_global_graph, batch_relevant_sentences = [], [], []
        batch_global_graph0, batch_global_graph1, batch_global_graph2, batch_global_graph3 = [], [], [], []
        # batch_local_graph_pmi = []
        batch_relevant_sentences, batch_relevant_sentences_presentation = [], []
        batch_pair_grid_labels, batch_triple_grid_labels = [], []
        # batch_aspect_grid_labels, batch_opinion_grid_labels = [], []
        batch_aspect_sequence_labels, batch_opinion_sequence_labels, batch_sentiment_sequence_labels = [], [], []
        batch_aspect_span_labels, batch_opinion_span_labels = [], []
        batch_train_sentences = []
        # 最大sentence0长度
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        # 最大relevant sentence个数
        max_sen_num = max([len(t['relevant_sentences']) for t in batch_data])
        max_train_len = max([len(t) for t in batch_data[0]['train_sentences_indices']])
        # 最大relevant sentence长度
        max_relevant_sen_len = 0
        for batch in batch_data:
            sentences = batch['relevant_sentence_presentation']
            for sentence in sentences:
                tem = len(sentence)
                if tem > max_relevant_sen_len:
                    max_relevant_sen_len = tem

        for item in batch_data:
            # read data
            text_indices, mask = item['text_indices'], item['mask']
            relevant_sentences, relevant_sentence_presentation = item['relevant_sentences'], item['relevant_sentence_presentation']
            global_graph0, global_graph1, global_graph2, global_graph3 = item['global_graph0'], item['global_graph1'], item['global_graph2'], item['global_graph3']
            aspect_sequence_label, opinion_sequence_label, sentiment_sequence_label = \
                item['aspect_sequence_label'], item['opinion_sequence_label'], item['sentiment_sequence_label']
            aspect_span_label, opinion_span_label = item['aspect_span_labels'], item['opinion_span_labels']
            pair_grid_label, triple_grid_label = item['pair_grid_labels'], item['triple_grid_labels']
            train_sentences_indices = item['train_sentences_indices']
            # aspect_grid_label, opinion_grid_label = item['aspect_grid_label'], item['opinion_grid_label']
            # pad all train sentences indices
            for i in range(len(train_sentences_indices)):
                train_sentences_indices[i] = train_sentences_indices[i] + [0] * (max_train_len - len(train_sentences_indices[i]))
            # padding relevant sentence_presentation
            relevant_sentence_presentation_ = []
            for re_sentence in relevant_sentence_presentation:
                temm = re_sentence
                for t in range(max_relevant_sen_len - len(re_sentence)):
                    temm.append(0)
                relevant_sentence_presentation_.append(temm)

            for jj in range(max_sen_num-len(relevant_sentence_presentation)):
                relevant_sentence_presentation_.append([0]*max_relevant_sen_len)

            # prepare for padding
            index_padding = [0] * (max_len - len(text_indices))
            text_padding = [-1] * (max_len - len(text_indices))
            sen_padding = [-1] * (max_sen_num - len(relevant_sentences))

            # generate aspect mask
            aspect_mask = []
            for i in aspect_sequence_label:
                if i != 0:
                    aspect_mask.append(1)
                else:
                    aspect_mask.append(0)
            # padding for global graph0
            global_graph0 = global_graph0.tolist()
            for i in range(len(global_graph0)):
                for j in range(max_sen_num-len(global_graph0[i])):
                    global_graph0[i].append(0)
            for i in range(max_len - len(text_indices)):
                global_graph0.append([0]*max_sen_num)
            global_graph0 = np.array(global_graph0)
            # padding for global graph 1
            global_graph1_= []
            for graph in global_graph1:
                global_graph1_.append(graph.tolist())
            for i in range(len(global_graph1_)):
                for j in range(len(global_graph1_[i])):
                    for k in range(max_relevant_sen_len - len(global_graph1_[i][j])):
                        global_graph1_[i][j].append(0)
            for i in range(len(global_graph1_)):
                for k in range(max_sen_num - len(global_graph1_[i])):
                    global_graph1_[i].append([0]*max_relevant_sen_len)
            global_graph1_ = np.array(global_graph1_)
            tem_graph = np.zeros_like(global_graph1_[0])
            tem_len = len(global_graph1_)
            for k in range(max_sen_num-tem_len):
                global_graph1_ = np.append(global_graph1_, [tem_graph], axis=0)
            # padding for global graph 2
            global_graph2_ = []
            for graph in global_graph2:
                global_graph2_.append(graph.tolist())
            for i in range(len(global_graph2_)):
                for j in range(len(global_graph2_[i])):
                    for k in range(max_relevant_sen_len - len(global_graph2_[i][j])):
                        global_graph2_[i][j].append(0)
            for i in range(len(global_graph2_)):
                for k in range(max_len - len(global_graph2_[i])):
                    global_graph2_[i].append([0]*max_relevant_sen_len)
            global_graph2_ = np.array(global_graph2_)
            tem_graph = np.zeros_like(global_graph2_[0])
            tem_len = len(global_graph2_)
            for k in range(max_sen_num-tem_len):
                global_graph2_ = np.append(global_graph2_, [tem_graph], axis=0)
            # padding for global graph 2
            global_graph3_ = []
            for graph in global_graph3:
                global_graph3_.append(graph.tolist())
            for i in range(len(global_graph3_)):
                for j in range(len(global_graph3_[i])):
                    for k in range(max_relevant_sen_len - len(global_graph3_[i][j])):
                        global_graph3_[i][j].append(0)
            for i in range(len(global_graph3_)):
                for k in range(max_len - len(global_graph3_[i])):
                    global_graph3_[i].append([0]*max_relevant_sen_len)
            global_graph3_ = np.array(global_graph3_)
            tem_graph = np.zeros_like(global_graph3_[0])
            tem_len = len(global_graph3_)
            for k in range(max_sen_num-tem_len):
                global_graph3_ = np.append(global_graph3_, [tem_graph], axis=0)
            # padding for pair_grid_labels
            pair_grid_label = pair_grid_label.tolist()
            for i in range(len(pair_grid_label)):
                for j in range(max_len - len(text_indices)):
                    pair_grid_label[i].append(-1)
            for i in range(max_len - len(text_indices)):
                pair_grid_label.append([-1]*max_len)
            pair_grid_label = np.array(pair_grid_label)
            # padding for triple_grid_label
            triple_grid_label = triple_grid_label.tolist()
            for i in range(len(triple_grid_label)):
                for j in range(max_len - len(text_indices)):
                    triple_grid_label[i].append(-1)
            for i in range(max_len - len(text_indices)):
                triple_grid_label.append([-1]*max_len)
            triple_grid_label = np.array(triple_grid_label)
            # padding for aspect_span_label
            aspect_span_label = aspect_span_label.tolist()
            for i in range(len(aspect_span_label)):
                for j in range(max_len - len(text_indices)):
                    aspect_span_label[i].append(-1)
            for i in range(max_len - len(text_indices)):
                aspect_span_label.append([-1]*max_len)
            aspect_span_label = np.array(aspect_span_label)
            # padding for opinion_span_label
            opinion_span_label = opinion_span_label.tolist()
            for i in range(len(opinion_span_label)):
                for j in range(max_len - len(text_indices)):
                    opinion_span_label[i].append(-1)
            for i in range(max_len - len(text_indices)):
                opinion_span_label.append([-1]*max_len)
            opinion_span_label = np.array(opinion_span_label)

            batch_text_indices.append(text_indices + index_padding)
            batch_mask.append(mask + index_padding)
            batch_aspect_mask.append(aspect_mask + index_padding)
            batch_global_graph0.append(global_graph0)
            batch_global_graph1.append(global_graph1_)
            batch_global_graph2.append(global_graph2_)
            batch_global_graph3.append(global_graph3_)
            batch_relevant_sentences.append(relevant_sentences + sen_padding)
            batch_relevant_sentences_presentation.append(relevant_sentence_presentation_)
            batch_pair_grid_labels.append(pair_grid_label)
            batch_triple_grid_labels.append(triple_grid_label)
            batch_aspect_sequence_labels.append(aspect_sequence_label + text_padding)
            batch_opinion_sequence_labels.append(opinion_sequence_label + text_padding)
            batch_sentiment_sequence_labels.append(sentiment_sequence_label + text_padding)
            batch_aspect_span_labels.append(aspect_span_label)
            batch_opinion_span_labels.append(opinion_span_label)
            batch_train_sentences.append(train_sentences_indices)
        return {'text_indices': torch.tensor(batch_text_indices), \
                'mask': torch.tensor(batch_mask),\
                'aspect_mask': torch.tensor(batch_aspect_mask),\
                'global_graph0': torch.tensor(batch_global_graph0),\
                'global_graph1': torch.tensor(batch_global_graph1),\
                'global_graph2': torch.tensor(batch_global_graph2),\
                'global_graph3': torch.tensor(batch_global_graph3),\
                'relevant_sentences': torch.tensor(batch_relevant_sentences),\
                'relevant_sentences_presentation': torch.tensor(batch_relevant_sentences_presentation),\
                'pair_grid_labels':torch.tensor(batch_pair_grid_labels),\
                'triple_grid_labels':torch.tensor(batch_triple_grid_labels),\
                'aspect_sequence_labels':torch.tensor(batch_aspect_sequence_labels),\
                'opinion_sequence_labels':torch.tensor(batch_opinion_sequence_labels),\
                'sentiment_sequence_labels':torch.tensor(batch_sentiment_sequence_labels),\
                'aspect_span_labels':torch.tensor(batch_aspect_span_labels),\
                'opinion_span_labels':torch.tensor(batch_opinion_span_labels),\
                'train_sentences_indices':torch.tensor(batch_train_sentences),\
                }


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
