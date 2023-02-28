# # -*- coding: utf-8 -*-

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
# import pdb
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer
# from transformers import BertForTokenClassification
# from transformers import BertModel, BertTokenizer
# import numpy as np

# def generate_formal_adj(init_adj):
#     '''input: a simple adj with a size of (row, column)
#         output: a complete and formal adj with a size of (row+column, row+column)'''
#     batch, row, column = init_adj.shape
#     # up left matrix (batch, row, row)
#     lu = torch.tensor(np.zeros((batch, row, row)).astype('float32')).cuda()
#     # up right (batch, row, column)
#     ru = init_adj.cuda()
#     # down left (batch, column, row)
#     ld = init_adj.transpose(1, 2).cuda()
#     # down right (batch, column, column)
#     rd = torch.tensor(np.zeros((batch, column, column)).astype('float32')).cuda()
#     # up (batch, row, row+column)
#     up = torch.cat([lu.float(), ru.float()], -1).cuda()
#     # down (batch, column, row+column)
#     down = torch.cat([ld.float(), rd.float()], -1).cuda()
#     # final （batch, row+column, row+column）
#     final = torch.cat([up,down],1).cuda()
#     return final.cuda()

# def preprocess_adj(A):
#     '''
#     for batch data
#     Pre-process adjacency matrix
#     :param A: adjacency matrix
#     :return:
#     '''
#     # prepare
#     assert A.shape[-1] == A.shape[-2]
#     batch = A.shape[0]
#     num = A.shape[-1]
#     # generate eye
#     I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
#     # 
#     A_hat = A.cuda() + I
#     #
#     D_hat_diag = torch.sum(A_hat.cuda(), axis=-1)
#     # 
#     D_hat_diag_inv_sqrt = torch.pow(D_hat_diag.cuda(), -0.5)
#     # inf 
#     D_hat_diag_inv_sqrt = torch.where(torch.isinf(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
#     D_hat_diag_inv_sqrt = torch.where(torch.isnan(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
#     # 
#     tem_I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
#     D_hat_diag_inv_sqrt_ = D_hat_diag_inv_sqrt.unsqueeze(-1).repeat(1,1,num).cuda()
#     D_hat_inv_sqrt = D_hat_diag_inv_sqrt_ * tem_I
#     # 
#     return torch.matmul(torch.matmul(D_hat_inv_sqrt.cuda(), A_hat.cuda()), D_hat_inv_sqrt.cuda())

# class SequenceLabelForAO(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(SequenceLabelForAO, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         """
#         Args:
#             input_features: (bs, seq_len, h)
#         """
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         sub_output = self.hidden2tag_sub(features_tmp)
#         obj_output = self.hidden2tag_obj(features_tmp)
#         return sub_output, obj_output

# class SequenceLabelForAOS(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(SequenceLabelForAOS, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.hidden2tag_senti = nn.Linear(int(hidden_size / 2), self.tag_size+1)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         """
#         Args:
#             input_features: (bs, seq_len, h)
#         """
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         sub_output = self.hidden2tag_sub(features_tmp)
#         obj_output = self.hidden2tag_obj(features_tmp)
#         senti_output = self.hidden2tag_senti(features_tmp)
#         return sub_output, obj_output, senti_output

# class CustomizeSequenceLabelForAO(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(CustomizeSequenceLabelForAO, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_sub = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_obj = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.linear_a = nn.Linear(hidden_size, self.tag_size)
#         self.linear_o = nn.Linear(hidden_size, self.tag_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         """
#         Args:
#             input_features: (bs, seq_len, h)
#         """
#         # share
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         # ATE
#         features_tmp_a = self.hidden2tag_sub(input_features)
#         features_tmp_a = nn.ReLU()(features_tmp)
#         features_tmp_a = self.dropout(features_tmp)
#         # OTE
#         features_tmp_o = self.hidden2tag_obj(input_features)
#         features_tmp_o = nn.ReLU()(features_tmp)
#         features_tmp_o = self.dropout(features_tmp)
#         # cat 
#         features_for_a = torch.cat([features_tmp, features_tmp_a], -1)
#         features_for_o = torch.cat([features_tmp, features_tmp_o], -1)
#         # classifier
#         sub_output = self.linear_a(features_for_a)
#         obj_output = self.linear_a(features_for_o)

#         return sub_output, obj_output

# class SequenceLabelForTriple(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(SequenceLabelForTriple, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size+1)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         """
#         Args:
#             input_features: (bs, seq_len, h)
#         """
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         sub_output = self.hidden2tag_sub(features_tmp)
#         obj_output = self.hidden2tag_obj(features_tmp)
#         return sub_output, obj_output

# class MultiNonLinearClassifier(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(MultiNonLinearClassifier, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         features_output = self.hidden2tag(features_tmp)
#         return features_output

# class PairGeneration(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, features, bias=False):
#         super(PairGeneration, self).__init__() # 32,13,300   32,300,13
#         self.features = features
#         # self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(features, features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, text):
#         hidden = torch.matmul(text.float(), self.weight)
#         # print(hidden.shape)
#         # denom = torch.sum(adj, dim=2, keepdim=True) + 1
#         # adj = torch.tensor(adj, dtype=torch.float32)
#         hidden_ = torch.tensor(hidden, dtype=torch.float32)
#         # print(hidden_.shape)
#         output = torch.matmul(hidden_, hidden.permute(0,2,1))
#         # print(output.shape)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

# class PairGeneration0(nn.Module):
#     def __init__(self, features, bias=False):
#         super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
#         self.features = features
#         # self.out_features = out_features
#         # self.weight = nn.Parameter(torch.FloatTensor(features, features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, text):
#         hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
#         hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
#         output = torch.cat((hidden_1, hidden_2),-1)
#         return output

# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=False):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
#         if acti:
#             self.acti = nn.ReLU(inplace=True)
#         else:
#             self.acti = None
#     def forward(self, F):
#         output = self.linear(F)
#         if not self.acti:
#             return output
#         return self.acti(output)

# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes, p):
#         super(GCN, self).__init__()
#         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
#         # self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
#         self.dropout = nn.Dropout(p)

#     def forward(self, A, X):
#         X = self.dropout(X.float().cuda())
#         F = torch.matmul(A.cuda(), X.cuda())
#         F = self.gcn_layer1(F.cuda())
#         output = F
#         # F = self.dropout(F.cuda())
#         # F = torch.matmul(A, F.cuda())
#         # output = self.gcn_layer2(F.cuda())
#         return output

# # class GraphConvolution(nn.Module):
# #     """
# #     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
# #     """
# #     def __init__(self, in_features, out_features, bias=True):
# #         super(GraphConvolution, self).__init__()
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
# #         # if bias:
# #         #     self.bias = nn.Parameter(torch.FloatTensor(out_features))
# #         # else:
# #         #     self.register_parameter('bias', None)

# #     def forward(self, text, adj):
# #         hidden = torch.matmul(text, self.weight)
# #         denom = torch.sum(adj, dim=2, keepdim=True) + 1
# #         # adj = torch.tensor(adj)
# #         adj = torch.tensor(adj, dtype=torch.float32)
# #         # hidden = torch.tensor(hidden)
# #         hidden = torch.tensor(hidden, dtype=torch.float32)
# #         output = torch.matmul(adj.cuda(), hidden.cuda()) / denom.cuda()
# #         # print(output.shape)
# #         # print(self.bias.shape)
# #         # if self.bias is not None:
# #         #     return output + self.bias
# #         # else:
# #         return output

# class PairGeneration(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, features, bias=False):
#         super(PairGeneration, self).__init__() # 32,13,300   32,300,13
#         self.features = features
#         # self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(features, features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, text):
#         hidden = torch.matmul(text.float(), self.weight)
#         # print(hidden.shape)
#         # denom = torch.sum(adj, dim=2, keepdim=True) + 1
#         # adj = torch.tensor(adj, dtype=torch.float32)
#         hidden_ = torch.tensor(hidden, dtype=torch.float32)
#         # print(hidden_.shape)
#         output = torch.matmul(hidden_, hidden.permute(0,2,1))
#         # print(output.shape)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

# class PairGeneration0(nn.Module):
#     def __init__(self, features, bias=False):
#         super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
#         self.features = features
#         # self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(features, features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, text):
#         hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
#         hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
#         output = torch.cat((hidden_1, hidden_2),-1)
#         return output

# class BERT_GCN0(nn.Module):
#     def __init__(self, opt, pretrained_model='bert-base-uncased', freeze_bert = False):
#         super(BERT_GCN0, self).__init__()
#         # 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'
#         self.opt = opt
#         pretrained_model = opt.bert_type
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#         self.bert_model = BertModel.from_pretrained(pretrained_model)
#         if freeze_bert:
#             for param in self.bert_model.parameters():
#                 param.requires_grad = False
#         self.gcn = GCN(768, 300, 3, 0.3)
#         self.text_embed_dropout = nn.Dropout(0.3)
#         self.pairgeneration = PairGeneration0(1068)

#         self.fc_aspect = nn.Linear(1068, 3)
#         self.fc_opinion = nn.Linear(1068, 3)
#         self.fc_sentiment = nn.Linear(1068, 4)

#         self.fc_pair = nn.Linear(1068*2, 3)
#         self.fc_pair_sentiment = nn.Linear(1068*2, 4)

#         # new classifier 
#         self.aspect_classifier = MultiNonLinearClassifier(1068, 3, 0.5)
#         self.opinion_classifier = MultiNonLinearClassifier(1068, 3, 0.5)
#         self.aspect_opinion_classifier = SequenceLabelForAO(1068, 3, 0.5)
#         self.customize_aspect_opinion_classifier = CustomizeSequenceLabelForAO(1068, 3, 0.5)
#         self.aspect_opinion_sentiment_classifier = SequenceLabelForAOS(1068, 3, 0.5)

#         self.sentiment_classifier = MultiNonLinearClassifier(1068, 4, 0.5)

#         self.pair_classifier = MultiNonLinearClassifier(1068*2, 3, 0.5)
#         self.pair_sentiment_classifier =  MultiNonLinearClassifier(1068*2, 4, 0.5)
        
#         self.triple_classifier = SequenceLabelForTriple(1068*2, 3, 0.5)

#     def forward(self, inputs, mask):
#         '''
#         text_indices: (batch_size, sentence_len)
#         mask: (batch_size, sentence_len)
#         relevant_sentences: (batch_size, rele_sen_num)
#         relevant_sentences_presentation: (batch_size, rele_sen_num, rele_sen_len)
#         '''
#         # input
#         text_indices, mask, relevant_sentences, relevant_sentences_presentation = inputs
#         # prepare
#         batch_size = text_indices.shape[0]
#         sentence_len = text_indices.shape[1]
#         rele_sen_num = relevant_sentences.shape[1]
#         rele_sen_len = relevant_sentences_presentation.shape[-1]
#         # input sentnece s_0
#         word_embeddings = self.bert_model(text_indices, mask)[0]
#         text_out = self.text_embed_dropout(word_embeddings)
#         # K relevant sentences (b, k, n) -> (b*k, n, 768) -> (b*k, 768) -> (b, k, 768)
#         relevant_sentence_out = self.bert_model(relevant_sentences_presentation.view(-1, rele_sen_len))[0][:,0,:]
#         relevant_sentence_out = relevant_sentence_out.view(batch_size, rele_sen_num,-1)
#         # prepare formal graph and features for GCN
#         # formal_global_adj = generate_formal_adj(global_adj)
#         # norm_global_adj = preprocess_adj(formal_global_adj)
#         formal_global_features = torch.cat([text_out, relevant_sentence_out], 1)
#         # attention
#         attention_feature = formal_global_features
#         attention = torch.matmul(attention_feature, attention_feature.permute(0, 2, 1))
#         attention = F.softmax(attention, -1)
#         norm_global_adj = preprocess_adj(attention)
#         # graph convolution with global graph
#         global_text_out = self.gcn(norm_global_adj, formal_global_features)[:, :sentence_len, :]
#         # concatenate
#         unified_text = torch.cat([text_out.float(), global_text_out.float()], -1)
#         # AE and OE scores
#         mask_ = torch.unsqueeze(mask, -1)
#         aspect_probs = (self.fc_aspect(unified_text.float()*mask_)).contiguous().view(-1, 3)
#         opinion_probs = (self.fc_opinion(unified_text.float()*mask_)).contiguous().view(-1, 3)
#         # pair generation
#         pair_text = self.pairgeneration(unified_text)
#         # pair mask
#         pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,pair_text.shape[1],1)\
#                      + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,pair_text.shape[1])
#         pair_mask_ = pair_mask.view(-1,1)
#         pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
#         # pair scores
#         '''old: pair_probs = self.fc_pair(pair_text.float()*pair_mask_grid).contiguous().view(-1, 3)'''
#         # pair sentiment scores
#         '''old: pair_sentiment_probs = self.fc_pair_sentiment(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)'''
#         pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float())
#         pair_probs = pair_probs_.contiguous().view(-1, 3)
#         pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)

#         return aspect_probs, opinion_probs, pair_probs, pair_sentiment_probs





# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import BertForTokenClassification
from transformers import BertModel, BertTokenizer
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

class GCNforFeature_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, p):
        super(GCNforFeature_1, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X.float())
        F = torch.matmul(A, X)
        output = self.gcn_layer1(F)
        return output

class GCNforFeature_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, p):
        super(GCNforFeature_2, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        self.gcn_layer2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X.float())
        F = torch.matmul(A, X)
        F = self.gcn_layer1(F)

        F = self.dropout(F.float())
        F = torch.matmul(A, F)
        output = self.gcn_layer2(F)
        return output

class SequenceLabelForAO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForAO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output

class SequenceLabelForAOS(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForAOS, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_senti = nn.Linear(int(hidden_size / 2), self.tag_size+1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        senti_output = self.hidden2tag_senti(features_tmp)
        return sub_output, obj_output, senti_output

class CustomizeSequenceLabelForAO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(CustomizeSequenceLabelForAO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_obj = nn.Linear(hidden_size, int(hidden_size / 2))
        self.linear_a = nn.Linear(hidden_size, self.tag_size)
        self.linear_o = nn.Linear(hidden_size, self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        # share
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        # ATE
        features_tmp_a = self.hidden2tag_sub(input_features)
        features_tmp_a = nn.ReLU()(features_tmp)
        features_tmp_a = self.dropout(features_tmp)
        # OTE
        features_tmp_o = self.hidden2tag_obj(input_features)
        features_tmp_o = nn.ReLU()(features_tmp)
        features_tmp_o = self.dropout(features_tmp)
        # cat 
        features_for_a = torch.cat([features_tmp, features_tmp_a], -1)
        features_for_o = torch.cat([features_tmp, features_tmp_o], -1)
        # classifier
        sub_output = self.linear_a(features_for_a)
        obj_output = self.linear_a(features_for_o)

        return sub_output, obj_output

class SequenceLabelForTriple(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForTriple, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size+1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output

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
        # self.weight = nn.Parameter(torch.FloatTensor(features, features))
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
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        # self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
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

class BERT_GCN0(nn.Module):
    def __init__(self, opt, pretrained_model='bert-base-uncased', freeze_bert = False):
        super(BERT_GCN0, self).__init__()
        # 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'
        dim_dic = {'bert-base-uncased':768, 'bert-large-uncased':1024}
        self.opt = opt
        pretrained_model = opt.bert_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        self.bert_model_ = BertModel.from_pretrained(pretrained_model)
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for param in self.bert_model_.parameters():
                param.requires_grad = False
        self.gcn = GCN(dim_dic[pretrained_model], 300, 3, 0.3)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.pairgeneration = PairGeneration0(dim_dic[pretrained_model]+300)

        self.gcn0 = GCNforFeature_1(dim_dic[pretrained_model], 300, 0.5)
        self.gcn1 = GCNforFeature_1(dim_dic[pretrained_model], 300, 0.5)
        self.gcn2 = GCNforFeature_2(dim_dic[pretrained_model], 300, 300, 0.5)
        self.gcn3 = GCNforFeature_2(dim_dic[pretrained_model], 300, 300, 0.5)

        self.aspect_opinion_classifier = SequenceLabelForAO((dim_dic[pretrained_model]+300+300)*2, 3, 0.5)
        self.triple_classifier = SequenceLabelForTriple((dim_dic[pretrained_model]+300+300)*2, 3, 0.5)

        self.fc_aspect = nn.Linear(dim_dic[pretrained_model]+300, 3)
        self.fc_opinion = nn.Linear(dim_dic[pretrained_model]+300, 3)
        self.fc_sentiment = nn.Linear(dim_dic[pretrained_model]+300, 4)

        self.fc_pair = nn.Linear((dim_dic[pretrained_model]+300)*2, 3)
        self.fc_pair_sentiment = nn.Linear((dim_dic[pretrained_model]+300)*2, 4)

        # new classifier 
        # self.aspect_classifier = MultiNonLinearClassifier((dim_dic[pretrained_model]+300), 3, 0.5)
        # self.opinion_classifier = MultiNonLinearClassifier((dim_dic[pretrained_model]+300), 3, 0.5)
        # self.aspect_opinion_classifier = SequenceLabelForAO((dim_dic[pretrained_model]+300), 3, 0.5)
        # self.customize_aspect_opinion_classifier = CustomizeSequenceLabelForAO((dim_dic[pretrained_model]+300), 3, 0.5)
        # self.aspect_opinion_sentiment_classifier = SequenceLabelForAOS((dim_dic[pretrained_model]+300), 3, 0.5)

        # self.sentiment_classifier = MultiNonLinearClassifier((dim_dic[pretrained_model]+300), 4, 0.5)

        # self.pair_classifier = MultiNonLinearClassifier((dim_dic[pretrained_model]+300)*2, 3, 0.5)
        # self.pair_sentiment_classifier =  MultiNonLinearClassifier((dim_dic[pretrained_model]+300)*2, 4, 0.5)
        
        # self.triple_classifier = SequenceLabelForTriple((dim_dic[pretrained_model]+300)*2, 3, 0.5)

    def forward(self, inputs, mask):
        '''
        text_indices: (batch_size, sentence_len)
        mask: (batch_size, sentence_len)
        relevant_sentences: (batch_size, rele_sen_num)
        relevant_sentences_presentation: (batch_size, rele_sen_num, rele_sen_len)
        '''
        # input
        text_indices, mask, relevant_sentences, relevant_sentences_presentation = inputs
        # prepare
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        rele_sen_num = relevant_sentences.shape[1]
        rele_sen_len = relevant_sentences_presentation.shape[-1]
        # input sentnece s_0
        word_embeddings = self.bert_model(text_indices, mask)[0]
        text_out = self.text_embed_dropout(word_embeddings)
        # K relevant sentences (b, k, n) -> (b*k, n, 768) -> (b*k, 768) -> (b, k, 768)
        relevant_sentence_out = self.bert_model_(relevant_sentences_presentation.view(-1, rele_sen_len))[0][:,0,:]
        relevant_sentence_out = relevant_sentence_out.view(batch_size, rele_sen_num,-1)
        relevant_sentence_out = self.text_embed_dropout(relevant_sentence_out)
        # get local and global 
        attention_feature_local, attention_feature_global = text_out, relevant_sentence_out
        attention_local, attention_global = \
                                        torch.matmul(attention_feature_local, attention_feature_local.permute(0, 2, 1)), \
                                        torch.matmul(attention_feature_local, attention_feature_global.permute(0, 2, 1))
        attention_local, attention_global = F.softmax(attention_local, -1), F.softmax(attention_global, -1)
        formal_attention_local, formal_attention_global = generate_formal_adj(attention_local), generate_formal_adj(attention_global)
        norm_local_adj, norm_global_adj = preprocess_adj(formal_attention_local), preprocess_adj(formal_attention_global)
        # get features
        formal_global_features = torch.cat([text_out, relevant_sentence_out], 1)
        formal_local_features = torch.cat([text_out, text_out], 1)
        # gcn
        if self.opt.gcn_layers_in_graph0 == 1:
            local_text_out = self.gcn0(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        elif self.opt.gcn_layers_in_graph0 == 2:
            local_text_out = self.gcn2(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn3(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        # concatenate
        unified_text = torch.cat([text_out.float(), local_text_out.float(), global_text_out.float()], -1)
        # AE and OE scores
        aspect_probs, opinion_probs = self.aspect_opinion_classifier(pair_text.float())
        aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
        # mask_ = torch.unsqueeze(mask, -1)
        # aspect_probs = (self.fc_aspect(unified_text.float()*mask_)).contiguous().view(-1, 3)
        # opinion_probs = (self.fc_opinion(unified_text.float()*mask_)).contiguous().view(-1, 3)
        # pair generation
        pair_text = self.pairgeneration(unified_text)
        # pair scores 
        pair_probs_, triple_probs_ = self.triple_classifier(pair_text.float())
        pair_probs = pair_probs_.contiguous().view(-1, 3)
        triple_probs = triple_probs_.contiguous().view(-1, 4)
        # pair mask
        # pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,pair_text.shape[1],1)\
        #              + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,pair_text.shape[1])
        # pair_mask_ = pair_mask.view(-1,1)
        # pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
        # pair scores
        '''old: pair_probs = self.fc_pair(pair_text.float()*pair_mask_grid).contiguous().view(-1, 3)'''
        # pair sentiment scores
        '''old: pair_sentiment_probs = self.fc_pair_sentiment(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)'''
        # pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float())
        # pair_probs = pair_probs_.contiguous().view(-1, 3)
        # pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)

        return aspect_probs, opinion_probs, pair_probs, triple_probs