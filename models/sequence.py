# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # import pdb; pdb.set_trace()
        # adj = torch.tensor(adj)
        adj = torch.tensor(adj, dtype=torch.float32)
        # hidden = torch.tensor(hidden)
        hidden = torch.tensor(hidden, dtype=torch.float32)
        output = torch.matmul(adj, hidden) / denom
        # print(output.shape)
        # print(self.bias.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class PairGeneration(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, features, bias=False):
        super(PairGeneration, self).__init__() # 32,13,300   32,300,13
        self.features = features
        # self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(features, features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text):
        hidden = torch.matmul(text.float(), self.weight)
        # print(hidden.shape)
        # denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # adj = torch.tensor(adj, dtype=torch.float32)
        hidden_ = torch.tensor(hidden, dtype=torch.float32)
        # print(hidden_.shape)
        output = torch.matmul(hidden_, hidden.permute(0,2,1))
        # print(output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SEQ(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SEQ, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.gcn = GraphConvolution(600, 300)
        self.gcn_ = GraphConvolution(600, 300)
        self.sigmoid = nn.Sigmoid()
        self.text_embed_dropout = nn.Dropout(0.3)
        self.soft = nn.Softmax(dim=-1)
        self.pairgeneration = PairGeneration(600)
        self.fc_pair = nn.Linear(1, 3)
        self.fc_aspect = nn.Linear(600, 3)

    def forward(self, inputs, mask):
        
        text_indices, mask, local_adj, global_adj, relevant_sentences, relevant_sentences_presentation, m_, n_, local_adj_pmi,_,_,_ = inputs
        # print(text_indices.dtype, mask.dtype, local_adj.dtype, global_adj.dtype, relevant_sentences.dtype, relevant_sentences_presentation.dtype)
        # import pdb; pdb.set_trace()
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        text = self.text_embed_dropout(word_embeddings)
        text_out, (_, _) = self.lstm(text, text_len)

        relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
        sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
        sentence_embedding = self.embed(relevant_sentences_presentation)     
        sentence_text_ = self.text_embed_dropout(sentence_embedding)
        sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        ones = torch.ones_like(sentence_text_len)
        sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len))
        sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
        sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
        # import pdb; pdb.set_trace()
        
        local_text_out = self.gcn(text_out, local_adj_pmi)
        global_text_out = self.gcn_(sentence_text_out1, global_adj)

        unified_text = torch.cat([local_text_out, global_text_out], -1)

        aspect_probs = self.fc_aspect(unified_text.to(torch.float32).contiguous().view(-1, unified_text.shape[-1]))
        # pair_text = self.pairgeneration(unified_text)
        # pair_text = torch.unsqueeze(pair_text, -1)
        
        # pair_probs = self.fc_pair(pair_text.contiguous().view(-1, pair_text.shape[-1]))

        # pair_probs = F.relu(pair_probs)
        # pair_probs_ = pair_probs.argmax(dim=-1)
        # pair_probs_ = pair_probs_.view(text_indices.shape[0],text_indices.shape[1],text_indices.shape[1])

        # return pair_probs, pair_probs_
        return aspect_probs
