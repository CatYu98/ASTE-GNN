# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np

def generate_formal_adj(init_adj):
    '''input: a simple adj with a size of (row, column)
        output: a complete and formal adj with a size of (row+column, row+column)'''
    batch, row, column = init_adj.shape
    # up left matrix (batch, row, row)
    lu = torch.tensor(np.zeros((batch, row, row)).astype('float32')).cuda()
    # up right (batch, row, column)
    ru = init_adj.cuda()
    # down left (batch, column, row)
    ld = init_adj.transpose(1, 2).cuda()
    # down right (batch, column, column)
    rd = torch.tensor(np.zeros((batch, column, column)).astype('float32')).cuda()
    # up (batch, row, row+column)
    up = torch.cat([lu.float(), ru.float()], -1).cuda()
    # down (batch, column, row+column)
    down = torch.cat([ld.float(), rd.float()], -1).cuda()
    # final （batch, row+column, row+column）
    final = torch.cat([up,down],1).cuda()
    return final.cuda()

def generate_unified_adj(batch, n, k, m, graph0=None, graph1=None, graph2=None, graph3=None):
    ''' batch is batch_size, n is the length of sentence0, k is the number of relevant sentences, m is the length of relevant sentence. '''
    local_graph = torch.tensor(np.zeros((batch, n, n)).astype('float32')).cuda()
    # graph 0
    if graph0 == None:
        graph0 = torch.tensor(np.zeros((batch, n, k)).astype('float32')).cuda()
    else:
        graph0 = graph0.cuda()
    # graph 1
    if graph1 == None:
        graph1 = torch.tensor(np.zeros((batch, k, k*m)).astype('float32')).cuda()
    else:
        graph1 = graph1.cuda()
    # graph 2
    if graph2 == None:
        graph2 = torch.tensor(np.zeros((batch, n, k*m)).astype('float32')).cuda()
    else:
        graph2 = graph2.cuda()
    # graph 3
    if graph3 == None:
        graph3 = torch.tensor(np.zeros((batch, n, k*m)).astype('float32')).cuda()
    else:
        graph3 = graph3.cuda()
    if graph2 == None:
        word2word_graph = graph3
    else:
        word2word_graph = graph2
    # sub_graph1
    sub_graph1 = torch.cat([local_graph.float(), graph0.float(), word2word_graph.float()], -1).cuda()
    # sub_graph2
    sub_sub_graph = torch.tensor(np.zeros((batch, k, (n+k))).astype('float32')).cuda()
    sub_graph2 = torch.cat([sub_sub_graph.float(), graph1.float()], -1).cuda()
    # sub_graph3
    sub_graph3 = torch.tensor(np.zeros((batch, k*m, (n+k+k*m))).astype('float32')).cuda()

    final_graph = torch.cat([sub_graph1, sub_graph2, sub_graph3], 1).cuda()

    return final_graph

def preprocess_adj(A):
    '''
    for batch data
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    # prepare
    assert A.shape[-1] == A.shape[-2]
    batch = A.shape[0]
    num = A.shape[-1]
    # generate eye
    I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
    # 
    A_hat = A.cuda() + I
    #
    D_hat_diag = torch.sum(A_hat.cuda(), axis=-1)
    # 
    D_hat_diag_inv_sqrt = torch.pow(D_hat_diag.cuda(), -0.5)
    # inf 
    D_hat_diag_inv_sqrt = torch.where(torch.isinf(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
    D_hat_diag_inv_sqrt = torch.where(torch.isnan(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
    # 
    tem_I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
    D_hat_diag_inv_sqrt_ = D_hat_diag_inv_sqrt.unsqueeze(-1).repeat(1,1,num).cuda()
    D_hat_inv_sqrt = D_hat_diag_inv_sqrt_ * tem_I
    # 
    return torch.matmul(torch.matmul(D_hat_inv_sqrt.cuda(), A_hat.cuda()), D_hat_inv_sqrt.cuda())

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

class PairGeneration0(nn.Module):
    def __init__(self, features, bias=False):
        super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
        self.features = features
        # self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(features, features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text):
        hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
        hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
        output = torch.cat((hidden_1, hidden_2),-1)
        return output

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim, acti=False)
        self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X.float().cuda())
        F = torch.matmul(A.cuda(), X.cuda())
        F = self.gcn_layer1(F.cuda())
        output = F
        # F = self.dropout(F.cuda())
        # F = torch.matmul(A, F.cuda())
        # output = self.gcn_layer2(F.cuda())
        return output

class TS4(nn.Module):
    def __init__(self, embedding_matrix, opt):
        '''This is a unifeid graph model for graph0 to graph3'''
        super(TS4, self).__init__()
        self.use_graph0 = opt.use_graph0
        self.use_graph1 = opt.use_graph1
        self.use_graph2 = opt.use_graph2
        self.use_graph3 = opt.use_graph3
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.gcn0 = GCN(600, 300, 3, 0.3)
        self.gcn1 = GCN(600, 300, 3, 0.3)
        self.gcn2 = GCN(600, 300, 150, 0.3)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.pairgeneration = PairGeneration0(150)
        
        self.fc_aspect_h = nn.Linear(900, 450)
        self.fc_opinion_h = nn.Linear(900, 450)
        self.fc_sentiment_h = nn.Linear(900, 450)
        self.fc_pair_h = nn.Linear(1800, 900)
        self.fc_pair_sentiment_h = nn.Linear(1800, 900)

        self.fc_aspect = nn.Linear(450, 3)
        self.fc_opinion = nn.Linear(450, 3)
        self.fc_sentiment = nn.Linear(450, 4)
        self.fc_pair = nn.Linear(900, 3)
        self.fc_pair_sentiment = nn.Linear(900, 4)

        self.fc_s0 = nn.Linear(600, 200)
        self.fc_r_s = nn.Linear(600, 200)
        self.fc_r_s_w = nn.Linear(600, 200)

    def forward(self, inputs, mask):
        
        # input
        text_indices, mask, global_adj, global_adj1, global_adj2, global_adj3, relevant_sentences, relevant_sentences_presentation = inputs 
        # prepare 
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        rele_sen_num = relevant_sentences.shape[1]
        rele_sen_len = relevant_sentences_presentation.shape[-1]
        # process input
        # global_adj1 = torch.reshape(global_adj1, (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
        global_adj1 = torch.reshape(global_adj1.permute(0,2,1,3), (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
        global_adj2 = torch.reshape(global_adj2.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
        global_adj3 = torch.reshape(global_adj3.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
        # get unified graph
        graph0 = global_adj if self.use_graph0==1 else None
        graph1 = global_adj1 if self.use_graph1==1 else None
        graph2 = global_adj2 if self.use_graph2==1 else None
        graph3 = global_adj3 if self.use_graph3==1 else None
        unified_graph = generate_unified_adj(batch_size, sentence_len, rele_sen_num, rele_sen_len, graph0, graph1, graph2, graph3)
        # norm for unified graph adj
        norm_unified_graph = preprocess_adj(unified_graph)
        # get sentence mask
        mask_ = mask.view(-1,1)
        # prepare features
        '''get the features of sentence0'''
        # input sentnece s_0
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        text = self.text_embed_dropout(word_embeddings)
        text_out, (_, _) = self.lstm(text, text_len) # 32, 13, 600
        text_out_narrow = self.fc_s0(text_out)
        '''get the features of sentence1 to sentence_k'''
        # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
        relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
        sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
        sentence_embedding = self.embed(relevant_sentences_presentation)     
        sentence_text_ = self.text_embed_dropout(sentence_embedding)
        sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        ones = torch.ones_like(sentence_text_len)
        # sentence word features
        sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len))
        sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
        sentence_text_out_narrow = self.fc_r_s_w(sentence_text_out)
        # sentence features
        sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
        sentence_text_out1_narrow = self.fc_r_s(sentence_text_out1)
        # get unified features
        unified_features = torch.cat([text_out, sentence_text_out1, torch.reshape(sentence_text_out, (batch_size, rele_sen_num*rele_sen_len, -1))], 1)        
        # features after GCN
        gcn_features = self.gcn0(norm_unified_graph, unified_features)[:, :sentence_len, :]
        # final features for classification
        final_features = torch.cat([text_out, gcn_features], -1)
        # AE and OE scores
        aspect_probs = self.fc_aspect(self.fc_aspect_h(final_features.float())).contiguous().view(-1, 3)
        opinion_probs = self.fc_opinion(self.fc_opinion_h(final_features.float())).contiguous().view(-1, 3)
        sentiment_probs = self.fc_sentiment(self.fc_sentiment_h(final_features.float())).contiguous().view(-1, 4)
        # aspect_probs = self.gcn0(norm_unified_graph, unified_features)[:, :sentence_len, :].contiguous().view(-1, 3)
        # opinion_probs = self.gcn1(norm_unified_graph, unified_features)[:, :sentence_len, :].contiguous().view(-1, 3)
        # pair mask
        pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
                     + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,text_out.shape[1])
        pair_mask_ = pair_mask.view(-1,1)
        # pair scores  
        # pair_hidden = self.gcn2(norm_unified_graph, unified_features)[:, :sentence_len, :]
        # pair generation
        pair_text = self.pairgeneration(final_features)
        pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
        pair_probs = self.fc_pair(self.fc_pair_h(pair_text * pair_mask_grid)).contiguous().view(-1, 3)
        pair_sentiment_probs = self.fc_pair_sentiment(self.fc_pair_sentiment_h(pair_text * pair_mask_grid)).contiguous().view(-1, 4)

        return aspect_probs, opinion_probs, sentiment_probs, pair_probs, pair_sentiment_probs