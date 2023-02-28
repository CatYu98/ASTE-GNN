# # -*- coding: utf-8 -*-

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
# import pdb
# import torch.nn.functional as F
# from torch.nn import init
# import numpy as np
# import math
# from torch import Tensor


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
#     A_hat = A.cuda() + I # - I
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
# # class SequenceLabelForAO(nn.Module):
# #     def __init__(self, hidden_size, tag_size, dropout_rate):
# #         super(SequenceLabelForAO, self).__init__()
# #         self.tag_size = tag_size
# #         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
# #         self.linear_a = nn.Linear(hidden_size, int(hidden_size / 2))
# #         self.linear_o = nn.Linear(hidden_size, int(hidden_size / 2))
# #         self.hidden2tag_sub = nn.Linear(int(hidden_size), self.tag_size)
# #         self.hidden2tag_obj = nn.Linear(int(hidden_size), self.tag_size)
# #         self.dropout = nn.Dropout(dropout_rate)

#     # def forward(self, input_features):
#     #     """
#     #     Args:
#     #         input_features: (bs, seq_len, h)
#     #     """
#     #     # features_tmp = self.linear(input_features)
#     #     # features_tmp = nn.ReLU()(features_tmp)
#     #     # features_tmp = self.dropout(features_tmp)
#     #     features_a = self.linear_a(input_features)
#     #     features_a = self.dropout(features_a)
#     #     features_o = self.linear_o(input_features)
#     #     features_o = self.dropout(features_o)
#     #     # pair-attention
#     #     # for aspect 
#     #     aspect2opinion = torch.matmul(torch.norm(features_a, p=2, dim=-1, keepdim=True), torch.norm(features_o, p=2, dim=-1, keepdim=True).permute(0,2,1))
#     #     aspect2opinion_attention = F.softmax(aspect2opinion, -1)
#     #     aspect2opinion_embedding = torch.matmul(aspect2opinion_attention, features_o)
#     #     # for opinion
#     #     opinion2aspect = torch.matmul(torch.norm(features_o, p=2, dim=-1, keepdim=True), torch.norm(features_a, p=2, dim=-1, keepdim=True).permute(0,2,1))
#     #     opinion2aspect_attention = F.softmax(opinion2aspect, -1)
#     #     opinion2aspect_embedding = torch.matmul(opinion2aspect_attention, features_a)
#     #     # concat
#     #     features_a = torch.cat([features_a, aspect2opinion_embedding], -1)
#     #     features_o = torch.cat([features_o, opinion2aspect_embedding], -1)
#     #     # classifier
#     #     sub_output = self.hidden2tag_sub(features_a)
#     #     obj_output = self.hidden2tag_obj(features_o)
#     #     return sub_output, obj_output

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

# class SequenceLabelForGrid(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(SequenceLabelForGrid, self).__init__()
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

# class PairGeneration(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, features: int, bias: bool=False, device=None, dtype=None):
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(PairGeneration, self).__init__()
#         self.features = features
#         self.weight1 = nn.Parameter(torch.empty((features, features), **factory_kwargs))
#         self.weight2 = nn.Parameter(torch.empty((features, features), **factory_kwargs))
#         if bias:
#             self.bias1 = nn.Parameter(torch.empty(features, **factory_kwargs))
#             self.bias2 = nn.Parameter(torch.empty(features, **factory_kwargs))
#         else:
#             self.register_parameter('bias1', None)
#             self.register_parameter('bias2', None)
#         self.reset_parameters()
#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
#         init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
#         if self.bias1 is not None and self.bias2 is not None:
#             fan_in1, _ = init._calculate_fan_in_and_fan_out(self.weight1)
#             fan_in2, _ = init._calculate_fan_in_and_fan_out(self.weight2)
#             bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
#             bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
#             init.uniform_(self.bias1, -bound1, bound1)
#             init.uniform_(self.bias2, -bound2, bound2)

#     def forward(self, input: Tensor) -> Tensor:
#         hidden1 = F.linear(input, self.weight1, self.bias1)
#         hidden2 = F.linear(input, self.weight2, self.bias2)
#         if self.bias1 is not None and self.bias2 is not None:
#             hidden1 = hidden1 + self.bias1
#             hidden2 = hidden2 + self.bias2
#         output = torch.matmul(hidden1, hidden2.permute(0, 2, 1))
#         return output

# class PairGeneration0(nn.Module):
#     # def __init__(self, features, bias=False):
#     def __init__(self):
#         super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
#         # self.features = features
#         # self.weight = nn.Parameter(torch.FloatTensor(features, features))
#         # if bias:
#         #     self.bias = nn.Parameter(torch.FloatTensor(features))
#         # else:
#         #     self.register_parameter('bias', None)

#     def forward(self, text):
#         hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
#         hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
#         output = torch.cat((hidden_1, hidden_2),-1)
#         return output

# class PairGeneration1(nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool=False, device = None, dtype = None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(PairGeneration1, self).__init__()
#         self.in_features, self.out_features = in_features, out_features
#         self.weight = nn.Parameter(torch.empty((out_features, in_features*2), **factory_kwargs))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
    
#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, text):
#         hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
#         hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
#         output = torch.cat((hidden_1, hidden_2),-1)
#         output = F.linear(output, self.weight, self.bias)
#         return output

# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim, bias=False) # bias = False is also ok.
#         if acti:
#             # self.acti = nn.ReLU(inplace=True)
#             self.acti = nn.PReLU()
#         else:
#             self.acti = None
#     def forward(self, F):
#         output = self.linear(F)
#         if not self.acti:
#             return output
#         return self.acti(output)

# class GCNforFeature_1(nn.Module):
#     def __init__(self, input_dim, hidden_dim, p):
#         super(GCNforFeature_1, self).__init__()
#         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(p)

#     def forward(self, A, X):
#         X = self.dropout(X.float())
#         F = torch.matmul(A, X)
#         output = self.gcn_layer1(F)
#         return output

# class GCNforFeature_2(nn.Module):
#     def __init__(self, input_dim, hidden_dim, out_dim, p):
#         super(GCNforFeature_2, self).__init__()
#         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
#         self.gcn_layer2 = GCNLayer(hidden_dim, out_dim)
#         self.dropout = nn.Dropout(p)

#     def forward(self, A, X):
#         X = self.dropout(X.float())
#         F = torch.matmul(A, X)
#         F = self.gcn_layer1(F)

#         F = self.dropout(F.float())
#         F = torch.matmul(A, F)
#         output = self.gcn_layer2(F)
#         return output

# class GCNforSequence(nn.Module):
#     def __init__(self, input_dim, hidden_dim, out_dim, p):
#         super(GCNforSequence, self).__init__()
#         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim, True)
#         self.gcn_layer2 = GCNLayer(hidden_dim, out_dim, False)
#         self.gcn_layer3 = GCNLayer(hidden_dim, out_dim, False)
#         self.dropout = nn.Dropout(p)

#     def forward(self, A, X):
#         X = self.dropout(X.float())
#         F = torch.matmul(A, X)
#         F = self.gcn_layer1(F)

#         F1 = self.dropout(F.float())
#         F1 = torch.matmul(A, F1)
#         output1 = self.gcn_layer2(F1)

#         F2 = self.dropout(F.float())
#         F2 = torch.matmul(A, F2)
#         output2 = self.gcn_layer3(F2)
#         return output1, output2

# class GCNforTriple(nn.Module):
#     def __init__(self, input_dim, hidden_dim, out_dim, class_num, p):
#         super(GCNforTriple, self).__init__()
#         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim, True)
#         self.gcn_layer2 = GCNLayer(hidden_dim, out_dim, False)
#         self.gcn_layer3 = GCNLayer(hidden_dim, out_dim, False)
#         self.pair_generation = PairGeneration0()
#         self.dropout = nn.Dropout(p)
#         self.linear1 = nn.Linear(out_dim*2, class_num, bias=False)
#         self.linear2 = nn.Linear(out_dim*2, class_num+1, bias=False)

#     def forward(self, A, X):
#         X = self.dropout(X.float())
#         F = torch.matmul(A, X)
#         F = self.gcn_layer1(F)

#         F1 = self.dropout(F.float())
#         F1 = torch.matmul(A, F1)
#         output1 = self.gcn_layer2(F1)
#         pair_text = self.pair_generation(output1)
#         # pair_text = pair_text[:, :sentence_len, :sentence_len, :]
#         # pair_probs = self.linear1(pair_text)

#         F2 = self.dropout(F.float())
#         F2 = torch.matmul(A, F2)
#         output2 = self.gcn_layer3(F2)
#         triple_text = self.pair_generation(output2)
#         # triple_text = triple_text[:, :sentence_len, :sentence_len, :]
#         # triple_probs = self.linear2(triple_text)
        
#         # return pair_probs, triple_probs
#         return pair_text, triple_text

# class TS0(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(TS0, self).__init__()
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.text_embed_dropout = nn.Dropout(0.5)
#         self.pairgeneration = PairGeneration0()
#         self.gridgeneration = PairGeneration(900)
#         self.pairgeneration1 = PairGeneration1(900, 900)

#         self.gcn0 = GCNforFeature_1(600, 300, 0.5)
#         self.gcn1 = GCNforFeature_1(600, 300, 0.5)
#         self.gcn2 = GCNforFeature_2(600, 300, 300, 0.5)

#         self.aspect_opinion_classifier = SequenceLabelForAO(900, 3, 0.5)
#         self.triple_classifier = SequenceLabelForTriple(1800, 3, 0.5)

#         self.aspect_opinion_sequence_classifier = GCNforSequence(600, 300, 3, 0.5)
#         self.pair_triple_classifier = GCNforTriple(600, 300, 150, 3, 0.5)

#         self.pair_classifier = nn.Linear(300, 3)
#         self.triplet_classifier = nn.Linear(300, 4)
#     def forward(self, inputs, mask):
#         # input
#         text_indices, mask, global_adj, relevant_sentences, relevant_sentences_presentation,_, _, _, _, _ = inputs
#         # prepare 
#         batch_size = text_indices.shape[0]
#         sentence_len = text_indices.shape[1]
#         rele_sen_num = relevant_sentences.shape[1]
#         rele_sen_len = relevant_sentences_presentation.shape[-1]
#         # process global adj to get formal adj and norm 
#         formal_global_adj = generate_formal_adj(global_adj)
#         norm_global_adj = preprocess_adj(formal_global_adj)
#         # get sentence mask
#         mask_ = mask.view(-1,1)
#         # input sentnece s_0
#         text_len = torch.sum(text_indices != 0, dim=-1)
#         word_embeddings = self.embed(text_indices)
#         text = self.text_embed_dropout(word_embeddings)
#         text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
#         # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
#         relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
#         sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
#         sentence_embedding = self.embed(relevant_sentences_presentation)     
#         sentence_text_ = self.text_embed_dropout(sentence_embedding)
#         sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
#         ones = torch.ones_like(sentence_text_len)
#         sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
#         sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
#         sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
#         attention = F.softmax(torch.matmul(sentence_text_out, sentence_text_out.permute(0,1,3,2)), dim=-1)
#         sentence_text_out2 = torch.matmul(attention, sentence_text_out).sum(2)
#         # process formal features to match the formal adj
#         formal_global_features = torch.cat([text_out, sentence_text_out1], 1)
#         # attention
#         attention_feature = formal_global_features
#         attention = torch.matmul(attention_feature, attention_feature.permute(0, 2, 1))
#         attention = F.softmax(attention, -1)
#         norm_global_adj = preprocess_adj(attention)
#         # GCN with local global graph
#         if self.opt.gcn_layers_in_graph0 == 1:
#             global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
#         elif self.opt.gcn_layers_in_graph0 == 2:
#             global_text_out = self.gcn2(norm_global_adj, formal_global_features)[:, :sentence_len, :]
#         # global_text_out_tem = torch.matmul(norm_global_adj, formal_global_features)[:, :sentence_len, :]
#         '''aspect_probs, opinion_probs = self.aspect_opinion_sequence_classifier(norm_global_adj, formal_global_features)
#         pair_text, triple_text = self.pair_triple_classifier(norm_global_adj, formal_global_features)
#         aspect_probs, opinion_probs = aspect_probs[:, :sentence_len, :], opinion_probs[:, :sentence_len, :]
#         pair_text, triple_text = pair_text[:, :sentence_len, :sentence_len, :], triple_text[:, :sentence_len, :sentence_len, :]
#         pair_probs, triple_probs = self.pair_classifier(pair_text), self.triplet_classifier(triple_text)'''
#         # import pdb; pdb.set_trace()
#         # unified features
#         unified_text = torch.cat([text_out.float(), global_text_out.float()], -1)
#         # pair generation
#         pair_text = self.pairgeneration(unified_text)
#         # grid_text = self.gridgeneration(unified_text)
#         # pair_text1 = self.pairgeneration1(unified_text)
#         # pair_mask 
#         # pair_mask = grid_text.unsqueeze(-1)
#         # pair_text = pair_mask * pair_text
#         # AE and OE scores (BIO tagging)
#         aspect_probs, opinion_probs = self.aspect_opinion_classifier(unified_text.float())
#         aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
#         # pair scores 
#         pair_probs_, triple_probs_ = self.triple_classifier(pair_text.float())
#         pair_probs = pair_probs_.contiguous().view(-1, 3)
#         triple_probs = triple_probs_.contiguous().view(-1, 4)
#         # import pdb; pdb.set_trace()
#         # return F.log_softmax(aspect_probs, dim=-1), F.log_softmax(opinion_probs, dim=-1), F.log_softmax(pair_probs, dim=-1), F.log_softmax(triple_probs, dim=-1)
#         # return F.softmax(aspect_probs, dim=-1), F.softmax(opinion_probs, dim=-1), F.softmax(pair_probs, dim=-1), F.softmax(triple_probs, dim=-1)
#         return aspect_probs, opinion_probs, pair_probs, triple_probs



# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math
from torch import Tensor


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
    A_hat = A.cuda() + I # - I
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
# class SequenceLabelForAO(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate):
#         super(SequenceLabelForAO, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.linear_a = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.linear_o = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag_sub = nn.Linear(int(hidden_size), self.tag_size)
#         self.hidden2tag_obj = nn.Linear(int(hidden_size), self.tag_size)
#         self.dropout = nn.Dropout(dropout_rate)

    # def forward(self, input_features):
    #     """
    #     Args:
    #         input_features: (bs, seq_len, h)
    #     """
    #     # features_tmp = self.linear(input_features)
    #     # features_tmp = nn.ReLU()(features_tmp)
    #     # features_tmp = self.dropout(features_tmp)
    #     features_a = self.linear_a(input_features)
    #     features_a = self.dropout(features_a)
    #     features_o = self.linear_o(input_features)
    #     features_o = self.dropout(features_o)
    #     # pair-attention
    #     # for aspect 
    #     aspect2opinion = torch.matmul(torch.norm(features_a, p=2, dim=-1, keepdim=True), torch.norm(features_o, p=2, dim=-1, keepdim=True).permute(0,2,1))
    #     aspect2opinion_attention = F.softmax(aspect2opinion, -1)
    #     aspect2opinion_embedding = torch.matmul(aspect2opinion_attention, features_o)
    #     # for opinion
    #     opinion2aspect = torch.matmul(torch.norm(features_o, p=2, dim=-1, keepdim=True), torch.norm(features_a, p=2, dim=-1, keepdim=True).permute(0,2,1))
    #     opinion2aspect_attention = F.softmax(opinion2aspect, -1)
    #     opinion2aspect_embedding = torch.matmul(opinion2aspect_attention, features_a)
    #     # concat
    #     features_a = torch.cat([features_a, aspect2opinion_embedding], -1)
    #     features_o = torch.cat([features_o, opinion2aspect_embedding], -1)
    #     # classifier
    #     sub_output = self.hidden2tag_sub(features_a)
    #     obj_output = self.hidden2tag_obj(features_o)
    #     return sub_output, obj_output

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

class SequenceLabelForGrid(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForGrid, self).__init__()
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

class PairGeneration(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, features: int, bias: bool=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PairGeneration, self).__init__()
        self.features = features
        self.weight1 = nn.Parameter(torch.empty((features, features), **factory_kwargs))
        self.weight2 = nn.Parameter(torch.empty((features, features), **factory_kwargs))
        if bias:
            self.bias1 = nn.Parameter(torch.empty(features, **factory_kwargs))
            self.bias2 = nn.Parameter(torch.empty(features, **factory_kwargs))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias1 is not None and self.bias2 is not None:
            fan_in1, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            fan_in2, _ = init._calculate_fan_in_and_fan_out(self.weight2)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            init.uniform_(self.bias1, -bound1, bound1)
            init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, input: Tensor) -> Tensor:
        hidden1 = F.linear(input, self.weight1, self.bias1)
        hidden2 = F.linear(input, self.weight2, self.bias2)
        if self.bias1 is not None and self.bias2 is not None:
            hidden1 = hidden1 + self.bias1
            hidden2 = hidden2 + self.bias2
        output = torch.matmul(hidden1, hidden2.permute(0, 2, 1))
        return output

class PairGeneration0(nn.Module):
    # def __init__(self, features, bias=False):
    def __init__(self):
        super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
        # self.features = features
        # self.weight = nn.Parameter(torch.FloatTensor(features, features))
        # if bias:
        #     self.bias = nn.Parameter(torch.FloatTensor(features))
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, text):
        hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
        hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
        output = torch.cat((hidden_1, hidden_2),-1)
        return output

class PairGeneration1(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=False, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PairGeneration1, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features*2), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text):
        hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
        hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
        output = torch.cat((hidden_1, hidden_2),-1)
        output = F.linear(output, self.weight, self.bias)
        return output

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False) # bias = False is also ok.
        if acti:
            # self.acti = nn.ReLU(inplace=True)
            self.acti = nn.PReLU()
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)

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

class GCNforSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, p):
        super(GCNforSequence, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim, True)
        self.gcn_layer2 = GCNLayer(hidden_dim, out_dim, False)
        self.gcn_layer3 = GCNLayer(hidden_dim, out_dim, False)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X.float())
        F = torch.matmul(A, X)
        F = self.gcn_layer1(F)

        F1 = self.dropout(F.float())
        F1 = torch.matmul(A, F1)
        output1 = self.gcn_layer2(F1)

        F2 = self.dropout(F.float())
        F2 = torch.matmul(A, F2)
        output2 = self.gcn_layer3(F2)
        return output1, output2

class GCNforTriple(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, class_num, p):
        super(GCNforTriple, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim, True)
        self.gcn_layer2 = GCNLayer(hidden_dim, out_dim, False)
        self.gcn_layer3 = GCNLayer(hidden_dim, out_dim, False)
        self.pair_generation = PairGeneration0()
        self.dropout = nn.Dropout(p)
        self.linear1 = nn.Linear(out_dim*2, class_num, bias=False)
        self.linear2 = nn.Linear(out_dim*2, class_num+1, bias=False)

    def forward(self, A, X):
        X = self.dropout(X.float())
        F = torch.matmul(A, X)
        F = self.gcn_layer1(F)

        F1 = self.dropout(F.float())
        F1 = torch.matmul(A, F1)
        output1 = self.gcn_layer2(F1)
        pair_text = self.pair_generation(output1)
        # pair_text = pair_text[:, :sentence_len, :sentence_len, :]
        # pair_probs = self.linear1(pair_text)

        F2 = self.dropout(F.float())
        F2 = torch.matmul(A, F2)
        output2 = self.gcn_layer3(F2)
        triple_text = self.pair_generation(output2)
        # triple_text = triple_text[:, :sentence_len, :sentence_len, :]
        # triple_probs = self.linear2(triple_text)
        
        # return pair_probs, triple_probs
        return pair_text, triple_text

class TS0(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TS0, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.text_embed_dropout = nn.Dropout(0.5)
        self.pairgeneration = PairGeneration0()
        self.gridgeneration = PairGeneration(900)
        self.pairgeneration1 = PairGeneration1(900, 900)

        self.gcn0 = GCNforFeature_1(600, 300, 0.5)
        self.gcn1 = GCNforFeature_1(600, 300, 0.5)
        self.gcn2 = GCNforFeature_2(600, 300, 300, 0.5)
        self.gcn3 = GCNforFeature_2(600, 300, 300, 0.5)

        self.aspect_opinion_classifier = SequenceLabelForAO(2400, 3, 0.5)
        self.triple_classifier = SequenceLabelForTriple(2400, 3, 0.5)

        self.aspect_opinion_sequence_classifier = GCNforSequence(600, 300, 3, 0.5)
        self.pair_triple_classifier = GCNforTriple(600, 300, 150, 3, 0.5)

        self.pair_classifier = nn.Linear(300, 3)
        self.triplet_classifier = nn.Linear(300, 4)
    # def forward(self, inputs, mask):
    #     # input
    #     text_indices, mask, global_adj, relevant_sentences, relevant_sentences_presentation,_, _, _, _, _ = inputs
    #     # prepare 
    #     batch_size = text_indices.shape[0]
    #     sentence_len = text_indices.shape[1]
    #     rele_sen_num = relevant_sentences.shape[1]
    #     rele_sen_len = relevant_sentences_presentation.shape[-1]
    #     # process global adj to get formal adj and norm 
    #     # formal_global_adj = generate_formal_adj(global_adj)
    #     # norm_global_adj = preprocess_adj(formal_global_adj)
    #     # get sentence mask
    #     mask_ = mask.view(-1,1)
    #     # input sentnece s_0
    #     text_len = torch.sum(text_indices != 0, dim=-1)
    #     word_embeddings = self.embed(text_indices)
    #     text = self.text_embed_dropout(word_embeddings)
    #     text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
    #     # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
    #     relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
    #     sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
    #     sentence_embedding = self.embed(relevant_sentences_presentation)     
    #     sentence_text_ = self.text_embed_dropout(sentence_embedding)
    #     sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
    #     ones = torch.ones_like(sentence_text_len)
    #     sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
    #     sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
    #     sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
    #     attention = F.softmax(torch.matmul(sentence_text_out, sentence_text_out.permute(0,1,3,2)), dim=-1)
    #     sentence_text_out2 = torch.matmul(attention, sentence_text_out).sum(2)
    #     # get local and global adj
    #     attention_feature_local, attention_feature_global = text_out, sentence_text_out1
    #     attention_local, attention_global = \
    #                                     torch.matmul(attention_feature_local, attention_feature_local.permute(0, 2, 1)), \
    #                                     torch.matmul(attention_feature_local, attention_feature_global.permute(0, 2, 1))
    #     attention_local, attention_global = F.softmax(attention_local, -1), F.softmax(attention_global, -1)
    #     formal_attention_local, formal_attention_global = generate_formal_adj(attention_local), generate_formal_adj(attention_global)
    #     norm_local_adj, norm_global_adj = preprocess_adj(formal_attention_local), preprocess_adj( formal_attention_global)
    #     # get features
    #     formal_global_features = torch.cat([text_out, sentence_text_out1], 1)
    #     formal_local_features = torch.cat([text_out, text_out], 1)
    #     # gcn
    #     if self.opt.gcn_layers_in_graph0 == 1:
    #         local_text_out = self.gcn0(norm_local_adj, formal_local_features)[:, :sentence_len, :]
    #         global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
    #     elif self.opt.gcn_layers_in_graph0 == 2:
    #         local_text_out = self.gcn2(norm_local_adj, formal_local_features)[:, :sentence_len, :]
    #         global_text_out = self.gcn3(norm_global_adj, formal_global_features)[:, :sentence_len, :]
    #     # unified features
    #     unified_text = torch.cat([text_out.float(), local_text_out.float(), global_text_out.float()], -1)
    #     # pair generation
    #     pair_text = self.pairgeneration(unified_text)
    #     # AE and OE scores (BIO tagging)
    #     aspect_probs, opinion_probs = self.aspect_opinion_classifier(pair_text.float())
    #     aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
    #     # pair scores 
    #     pair_probs_, triple_probs_ = self.triple_classifier(pair_text.float())
    #     pair_probs = pair_probs_.contiguous().view(-1, 3)
    #     triple_probs = triple_probs_.contiguous().view(-1, 4)
    #     return aspect_probs, opinion_probs, pair_probs, triple_probs
    def forward(self, inputs, mask):
        # input
        text_indices, mask, global_adj, relevant_sentences, relevant_sentences_presentation,_, _, _, _, _, train_sentences_indices = inputs
        import pdb; pdb.set_trace()
        # prepare 
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        rele_sen_num = relevant_sentences.shape[1]
        rele_sen_len = relevant_sentences_presentation.shape[-1]
        # process global adj to get formal adj and norm 
        # formal_global_adj = generate_formal_adj(global_adj)
        # norm_global_adj = preprocess_adj(formal_global_adj)
        # get sentence mask
        mask_ = mask.view(-1,1)
        # input sentnece s_0
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        # flat input all sentence and get embeddings according to glove 
        train_sentences_indices_ = torch.reshape(train_sentences_indices, (-1, train_sentences_indices.shape[-1]))
        train_sentence_text_len = torch.sum(train_sentences_indices_!=0, dim=-1)
        train_sentences_embeddings = self.embed(train_sentences_indices_)
        # dropout and reshape embedding to the init shape 
        train_sentences_embeddings_ = self.text_embed_dropout(train_sentences_embeddings)
        train_sentences_text = torch.reshape(train_sentences_embeddings_, (-1, train_sentences_embeddings_.shape[-2], train_sentences_embeddings_.shape[-1]))
        # lstm to encode train sentences
        ones = torch.ones_like(train_sentence_text_len)
        train_sentences_text_out, (train_sentences_text_out1, b_) = self.lstm_(train_sentences_text, torch.where(train_sentence_text_len <= 0, ones, train_sentence_text_len).cpu())
        train_sentences_text_out = torch.reshape(train_sentences_text_out, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], train_sentences_indices.shape[2], train_sentences_text_out.shape[-1]))
        train_sentences_text_out1 = torch.reshape(train_sentences_text_out1, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], -1))
        # lstm to encode text embedding
        text = self.text_embed_dropout(word_embeddings)
        text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
        # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
        # relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
        # sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
        # sentence_embedding = self.embed(relevant_sentences_presentation)     
        # sentence_text_ = self.text_embed_dropout(sentence_embedding)
        # sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
        # ones = torch.ones_like(sentence_text_len)
        # sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
        # sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
        # sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
        # attention = F.softmax(torch.matmul(sentence_text_out, sentence_text_out.permute(0,1,3,2)), dim=-1)
        # sentence_text_out2 = torch.matmul(attention, sentence_text_out).sum(2)
        # get local and global adj
        attention_feature_local, attention_feature_global = text_out, sentence_text_out1
        attention_local, attention_global = \
                                        torch.matmul(attention_feature_local, attention_feature_local.permute(0, 2, 1)), \
                                        torch.matmul(attention_feature_local, attention_feature_global.permute(0, 2, 1))
        attention_local, attention_global = F.softmax(attention_local, -1), F.softmax(attention_global, -1)
        formal_attention_local, formal_attention_global = generate_formal_adj(attention_local), generate_formal_adj(attention_global)
        norm_local_adj, norm_global_adj = preprocess_adj(formal_attention_local), preprocess_adj( formal_attention_global)
        # get features
        formal_global_features = torch.cat([text_out, sentence_text_out1], 1)
        formal_local_features = torch.cat([text_out, text_out], 1)
        # gcn
        if self.opt.gcn_layers_in_graph0 == 1:
            local_text_out = self.gcn0(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        elif self.opt.gcn_layers_in_graph0 == 2:
            local_text_out = self.gcn2(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn3(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        # unified features
        unified_text = torch.cat([text_out.float(), local_text_out.float(), global_text_out.float()], -1)
        # pair generation
        pair_text = self.pairgeneration(unified_text)
        # AE and OE scores (BIO tagging)
        aspect_probs, opinion_probs = self.aspect_opinion_classifier(pair_text.float())
        aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
        # pair scores 
        pair_probs_, triple_probs_ = self.triple_classifier(pair_text.float())
        pair_probs = pair_probs_.contiguous().view(-1, 3)
        triple_probs = triple_probs_.contiguous().view(-1, 4)
        return aspect_probs, opinion_probs, pair_probs, triple_probs