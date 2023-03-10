# # # -*- coding: utf-8 -*-

# # from layers.dynamic_rnn import DynamicLSTM
# # import torch
# # import torch.nn as nn
# # import pdb
# # import torch.nn.functional as F
# # import numpy as np

# # def generate_formal_adj(init_adj):
# #     '''input: a simple adj with a size of (row, column)
# #         output: a complete and formal adj with a size of (row+column, row+column)'''
# #     batch, row, column = init_adj.shape
# #     # up left matrix (batch, row, row)
# #     lu = torch.tensor(np.zeros((batch, row, row)).astype('float32')).cuda()
# #     # up right (batch, row, column)
# #     ru = init_adj.cuda()
# #     # down left (batch, column, row)
# #     ld = init_adj.transpose(1, 2).cuda()
# #     # down right (batch, column, column)
# #     rd = torch.tensor(np.zeros((batch, column, column)).astype('float32')).cuda()
# #     # up (batch, row, row+column)
# #     up = torch.cat([lu.float(), ru.float()], -1).cuda()
# #     # down (batch, column, row+column)
# #     down = torch.cat([ld.float(), rd.float()], -1).cuda()
# #     # final （batch, row+column, row+column）
# #     final = torch.cat([up,down],1).cuda()
# #     return final.cuda()

# # def preprocess_adj(A):
# #     '''
# #     for batch data
# #     Pre-process adjacency matrix
# #     :param A: adjacency matrix
# #     :return:
# #     '''
# #     # prepare
# #     assert A.shape[-1] == A.shape[-2]
# #     batch = A.shape[0]
# #     num = A.shape[-1]
# #     # generate eye
# #     I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
# #     # 
# #     A_hat = A.cuda() + I
# #     #
# #     D_hat_diag = torch.sum(A_hat.cuda(), axis=-1)
# #     # 
# #     D_hat_diag_inv_sqrt = torch.pow(D_hat_diag.cuda(), -0.5)
# #     # inf 
# #     D_hat_diag_inv_sqrt = torch.where(torch.isinf(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
# #     D_hat_diag_inv_sqrt = torch.where(torch.isnan(D_hat_diag_inv_sqrt.cuda()), torch.full_like(D_hat_diag_inv_sqrt.cuda(), 0), D_hat_diag_inv_sqrt.cuda())
# #     # 
# #     tem_I = torch.eye(num).unsqueeze(0).repeat(batch, 1, 1).cuda()
# #     D_hat_diag_inv_sqrt_ = D_hat_diag_inv_sqrt.unsqueeze(-1).repeat(1,1,num).cuda()
# #     D_hat_inv_sqrt = D_hat_diag_inv_sqrt_ * tem_I
# #     # 
# #     return torch.matmul(torch.matmul(D_hat_inv_sqrt.cuda(), A_hat.cuda()), D_hat_inv_sqrt.cuda())

# # class GraphConvolution(nn.Module):
# #     """
# #     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
# #     """
# #     def __init__(self, in_features, out_features, bias=True):
# #         super(GraphConvolution, self).__init__()
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
# #         if bias:
# #             self.bias = nn.Parameter(torch.FloatTensor(out_features))
# #         else:
# #             self.register_parameter('bias', None)

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
# #         if self.bias is not None:
# #             return output + self.bias
# #         else:
# #             return output

# # class PairGeneration(nn.Module):
# #     """
# #     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
# #     """
# #     def __init__(self, features, bias=False):
# #         super(PairGeneration, self).__init__() # 32,13,300   32,300,13
# #         self.features = features
# #         # self.out_features = out_features
# #         self.weight = nn.Parameter(torch.FloatTensor(features, features))
# #         if bias:
# #             self.bias = nn.Parameter(torch.FloatTensor(features))
# #         else:
# #             self.register_parameter('bias', None)

# #     def forward(self, text):
# #         hidden = torch.matmul(text.float(), self.weight)
# #         # print(hidden.shape)
# #         # denom = torch.sum(adj, dim=2, keepdim=True) + 1
# #         # adj = torch.tensor(adj, dtype=torch.float32)
# #         hidden_ = torch.tensor(hidden, dtype=torch.float32)
# #         # print(hidden_.shape)
# #         output = torch.matmul(hidden_, hidden.permute(0,2,1))
# #         # print(output.shape)
# #         if self.bias is not None:
# #             return output + self.bias
# #         else:
# #             return output

# # class PairGeneration0(nn.Module):
# #     def __init__(self, features, bias=False):
# #         super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
# #         self.features = features
# #         # self.out_features = out_features
# #         self.weight = nn.Parameter(torch.FloatTensor(features, features))
# #         if bias:
# #             self.bias = nn.Parameter(torch.FloatTensor(features))
# #         else:
# #             self.register_parameter('bias', None)

# #     def forward(self, text):
# #         hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
# #         hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
# #         output = torch.cat((hidden_1, hidden_2),-1)
# #         return output

# # class GCNLayer(nn.Module):
# #     def __init__(self, in_dim, out_dim, acti=False):
# #         super(GCNLayer, self).__init__()
# #         self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
# #         if acti:
# #             self.acti = nn.ReLU(inplace=True)
# #         else:
# #             self.acti = None
# #     def forward(self, F):
# #         output = self.linear(F)
# #         if not self.acti:
# #             return output
# #         return self.acti(output)

# # class GCN(nn.Module):
# #     def __init__(self, input_dim, hidden_dim, num_classes, p):
# #         super(GCN, self).__init__()
# #         self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
# #         self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
# #         self.dropout = nn.Dropout(p)

# #     def forward(self, A, X):
# #         X = self.dropout(X.float().cuda())
# #         F = torch.matmul(A.cuda(), X.cuda())
# #         F = self.gcn_layer1(F.cuda())
# #         output = F
# #         # F = self.dropout(F.cuda())
# #         # F = torch.matmul(A, F.cuda())
# #         # output = self.gcn_layer2(F.cuda())
# #         return output

# # class TS3(nn.Module):
# #     def __init__(self, embedding_matrix, opt):
# #         super(TS3, self).__init__()
# #         self.opt = opt
# #         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
# #         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
# #         self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
# #         self.gcn0 = GCN(200, 100, 3, 0.3)
# #         self.gcn1 = GCN(200, 200, 3, 0.3)
# #         self.gcn2 = GCN(200, 100, 3, 0.3)
# #         self.sigmoid = nn.Sigmoid()
# #         self.text_embed_dropout = nn.Dropout(0.3)
# #         self.soft = nn.Softmax(dim=-1)
# #         self.pairgeneration = PairGeneration0(600)

# #         self.fc_aspect_h = nn.Linear(400, 200)
# #         self.fc_opinion_h = nn.Linear(400, 200)
# #         self.fc_sentiment_h = nn.Linear(400, 200)
# #         self.fc_pair_h = nn.Linear(800, 400)
# #         self.fc_pair_sentiment_h = nn.Linear(800, 400)

# #         self.fc_aspect = nn.Linear(200, 3)
# #         self.fc_opinion = nn.Linear(200, 3)
# #         self.fc_sentiment = nn.Linear(200, 4)
# #         self.fc_pair = nn.Linear(400, 3)
# #         self.fc_pair_sentiment = nn.Linear(400, 4)

# #         self.fc_s0 = nn.Linear(600, 200)
# #         self.fc_r_s = nn.Linear(600, 200)
# #         self.fc_r_s_w = nn.Linear(600, 200)

# #     def forward(self, inputs, mask):
        
# #         # input
# #         text_indices, mask, local_adj, global_adj, global_adj1, global_adj2, relevant_sentences, relevant_sentences_presentation, m_, n_, local_adj_pmi,_,_,_ = inputs 
# #         # prepare 
# #         batch_size = text_indices.shape[0]
# #         sentence_len = text_indices.shape[1]
# #         rele_sen_num = relevant_sentences.shape[1]
# #         rele_sen_len = relevant_sentences_presentation.shape[-1]
# #         # process input
# #         # global_adj1 = torch.reshape(global_adj1, (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
# #         global_adj1 = torch.reshape(global_adj1.permute(0,2,1,3), (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
# #         global_adj2 = torch.reshape(global_adj2.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
# #         # process global adj to get formal adj and norm 
# #         # graph 0
# #         formal_global_adj = generate_formal_adj(global_adj)
# #         norm_global_adj = preprocess_adj(formal_global_adj)
# #         # graph 1
# #         formal_global_adj1 = generate_formal_adj(global_adj1)
# #         norm_global_adj1 = preprocess_adj(formal_global_adj1)
# #         # graph 2
# #         formal_global_adj2 = generate_formal_adj(global_adj2)
# #         norm_global_adj2 = preprocess_adj(formal_global_adj2)
# #         # get sentence mask
# #         mask_ = mask.view(-1,1)
# #         '''get the features of sentence0'''
# #         # input sentnece s_0
# #         text_len = torch.sum(text_indices != 0, dim=-1)
# #         word_embeddings = self.embed(text_indices)
# #         text = self.text_embed_dropout(word_embeddings)
# #         text_out, (_, _) = self.lstm(text, text_len) # 32, 13, 600
# #         text_out_narrow = self.fc_s0(text_out)
# #         '''get the features of sentence1 to sentence_k'''
# #         # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
# #         relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
# #         sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
# #         sentence_embedding = self.embed(relevant_sentences_presentation)     
# #         sentence_text_ = self.text_embed_dropout(sentence_embedding)
# #         sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
# #         ones = torch.ones_like(sentence_text_len)
# #         # sentence word features
# #         sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len))
# #         sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
# #         sentence_text_out_narrow = self.fc_r_s_w(sentence_text_out)
# #         # sentence features
# #         sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
# #         sentence_text_out1_narrow = self.fc_r_s(sentence_text_out1)
# #         '''graph 2'''
# #         # process formal features to match the formal adj; first row then column
# #         # global graph: row -> relevant sentence feature, column -> relevant sentence word feature
# #         relevant_sentence_word_features = torch.reshape(sentence_text_out_narrow, (batch_size, rele_sen_num*rele_sen_len, -1))
# #         sentence0_sentence_word_features = text_out_narrow
# #         formal_global_features2 = torch.cat([sentence0_sentence_word_features, relevant_sentence_word_features], 1)
# #         # GCN with local global graph1 to get relevant sentences features
# #         global_text_out2 = self.gcn2(norm_global_adj2, formal_global_features2)[:, :sentence_len, :]
# #         '''graph 1'''
# #         # process formal features to match the formal adj; first row then column
# #         # global graph: row -> relevant sentence feature, column -> relevant sentence word feature
# #         relevant_sentence_features = sentence_text_out1_narrow
# #         relevant_sentence_word_features = torch.reshape(sentence_text_out_narrow, (batch_size, rele_sen_num*rele_sen_len, -1))
# #         formal_global_features1 = torch.cat([relevant_sentence_features, relevant_sentence_word_features], 1)
# #         # GCN with global graph1 to get relevant sentences features
# #         global_text_out1 = self.gcn1(norm_global_adj1, formal_global_features1)[:, :rele_sen_num, :]
# #         '''graph 0'''
# #         # process formal features to match the formal adj
# #         # pdb.set_trace()
# #         formal_global_features = torch.cat([text_out_narrow, global_text_out1], 1)
# #         # GCN with local global graph
# #         global_text_out = self.gcn0(norm_global_adj, formal_global_features)[:, :sentence_len, :]
# #         # pdb.set_trace()
# #         '''get features for final classification'''
# #         # unified features
# #         unified_text = torch.cat([text_out_narrow.float(), global_text_out.float(),  global_text_out2.float()], -1)
# #         # pair generation
# #         pair_text = self.pairgeneration(unified_text)
# #         # AE and OE scores
# #         aspect_probs = self.fc_aspect(self.fc_aspect_h(unified_text.float())).contiguous().view(-1, 3)
# #         opinion_probs = self.fc_opinion(self.fc_opinion_h(unified_text.float())).contiguous().view(-1, 3)
# #         sentiment_probs = self.fc_sentiment(self.fc_sentiment_h(unified_text.float())).contiguous().view(-1, 4)

# #         # pair mask
# #         pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
# #                      + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,text_out.shape[1])
# #         pair_mask_ = pair_mask.view(-1,1)
# #         pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])

# #         # pair scores     
# #         pair_probs = self.fc_pair(self.fc_pair_h(pair_text.float()*pair_mask_grid)).contiguous().view(-1, 3)

# #         # pair sentiment scores     
# #         pair_sentiment_probs = self.fc_pair_sentiment(self.fc_pair_sentiment_h(pair_text.float()*pair_mask_grid)).contiguous().view(-1, 4)

# #         return aspect_probs, opinion_probs, sentiment_probs, pair_probs, pair_sentiment_probs


# # -*- coding: utf-8 -*-

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
# import pdb
# import torch.nn.functional as F
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

# def pairgeneration(text):
#     hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
#     hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
#     output = hidden_1 + hidden_2
#     return output

# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
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
#         output = self.gcn_layer1(F)
#         return output

# class TS3(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(TS3, self).__init__()
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.text_embed_dropout = nn.Dropout(0.5)
#         self.pairgeneration = PairGeneration0(900)

#         self.gcn_local = GCNforFeature_1(600, 300, 0.5)
#         self.gcn0 = GCNforFeature_1(600, 300, 0.5)
#         self.gcn1 = GCNforFeature_1(600, 300, 0.5)
#         self.gcn2 = GCNforFeature_2(600, 300, 150, 0.5)

#         self.aspect_opinion_classifier = SequenceLabelForAO(900, 3, 0.5)
        
#         self.triple_classifier = SequenceLabelForTriple(1800, 3, 0.5)
#     def forward(self, inputs, mask):
        
#         # input
#         text_indices, mask, global_adj, global_adj1, global_adj2, relevant_sentences, relevant_sentences_presentation, m_, n_, _,_,_ = inputs 
#         # prepare 
#         batch_size = text_indices.shape[0]
#         sentence_len = text_indices.shape[1]
#         rele_sen_num = relevant_sentences.shape[1]
#         rele_sen_len = relevant_sentences_presentation.shape[-1]
#         # process local adj to get formal adj
#         # norm_local_adj = preprocess_adj(local_adj).float()
#         # process input
#         # global_adj1 = torch.reshape(global_adj1, (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
#         global_adj1 = torch.reshape(global_adj1.permute(0,2,1,3), (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
#         global_adj2 = torch.reshape(global_adj2.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
#         # process global adj to get formal adj and norm 
#         # graph 0
#         formal_global_adj = generate_formal_adj(global_adj)
#         norm_global_adj = preprocess_adj(formal_global_adj)
#         # graph 1
#         formal_global_adj1 = generate_formal_adj(global_adj1)
#         norm_global_adj1 = preprocess_adj(formal_global_adj1)
#         # graph 2
#         formal_global_adj2 = generate_formal_adj(global_adj2)
#         norm_global_adj2 = preprocess_adj(formal_global_adj2)
#         # get sentence mask
#         mask_ = mask.view(-1,1)
#         '''get the features of sentence0'''
#         # input sentnece s_0
#         text_len = torch.sum(text_indices != 0, dim=-1)
#         word_embeddings = self.embed(text_indices)
#         text = self.text_embed_dropout(word_embeddings)
#         text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
#         # use local graph
#         # local_text_out = self.gcn_local(norm_local_adj, text_out)
#         # text_out_narrow = self.fc_s0(text_out)
#         '''get the features of sentence1 to sentence_k'''
#         # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
#         relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
#         sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
#         sentence_embedding = self.embed(relevant_sentences_presentation_)     
#         # sentence_text_ = self.text_embed_dropout(sentence_embedding)
#         # sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
#         ones = torch.ones_like(sentence_text_len)
#         # sentence word features
#         sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_embedding, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
#         sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
#         # sentence_text_out_narrow = self.fc_r_s_w(sentence_text_out)
#         # sentence features
#         sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
#         # sentence_text_out1_narrow = self.fc_r_s(sentence_text_out1)
#         '''graph 2'''
#         # process formal features to match the formal adj; first row then column
#         # global graph: row -> relevant sentence feature, column -> relevant sentence word feature
#         relevant_sentence_word_features = torch.reshape(sentence_text_out, (batch_size, rele_sen_num*rele_sen_len, -1))
#         sentence0_sentence_word_features = text_out
#         formal_global_features2 = torch.cat([sentence0_sentence_word_features, relevant_sentence_word_features], 1)
#         # attention
#         attention_feature2 = formal_global_features2
#         attention2 = torch.matmul(attention_feature2, attention_feature2.permute(0, 2, 1))
#         attention2 = F.softmax(attention2, -1)
#         # norm_global_adj2 = preprocess_adj(attention2*torch.where(formal_global_adj2>0.8, torch.ones_like(formal_global_adj2), torch.zeros_like(formal_global_adj2)))
#         norm_global_adj2 = preprocess_adj(attention2)
#         # GCN with local global graph1 to get sentence0 word features
#         global_text_out2 = self.gcn1(norm_global_adj2, formal_global_features2)[:, :sentence_len, :]
#         '''graph 0'''
#         # process formal features to match the formal adj
#         formal_global_features = torch.cat([text_out, sentence_text_out1], 1)
#         # GCN with local global graph
#         global_text_out = self.gcn0(norm_global_adj, formal_global_features)[:, :sentence_len, :]

#         # unified features
#         unified_text = torch.cat([text_out.float(), global_text_out2.float()], -1)
#         # pair generation
#         pair_text = self.pairgeneration(unified_text)
#         # AE and OE scores
#         aspect_probs, opinion_probs = self.aspect_opinion_classifier(unified_text.float())
#         aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
#         # pair scores 
#         pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float())
#         pair_probs = pair_probs_.contiguous().view(-1, 3)
#         pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)   


#         return aspect_probs, opinion_probs, pair_probs, pair_sentiment_probs



# # -*- coding: utf-8 -*-

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
# import pdb
# import torch.nn.functional as F
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

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, text, adj):
#         hidden = torch.matmul(text, self.weight)
#         denom = torch.sum(adj, dim=2, keepdim=True) + 1
#         # adj = torch.tensor(adj)
#         adj = torch.tensor(adj, dtype=torch.float32)
#         # hidden = torch.tensor(hidden)
#         hidden = torch.tensor(hidden, dtype=torch.float32)
#         output = torch.matmul(adj.cuda(), hidden.cuda()) / denom.cuda()
#         # print(output.shape)
#         # print(self.bias.shape)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

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
#         self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
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

# class TS3(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(TS3, self).__init__()
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.gcn0 = GCN(200, 100, 3, 0.3)
#         self.gcn1 = GCN(200, 200, 3, 0.3)
#         self.gcn2 = GCN(200, 100, 3, 0.3)
#         self.sigmoid = nn.Sigmoid()
#         self.text_embed_dropout = nn.Dropout(0.3)
#         self.soft = nn.Softmax(dim=-1)
#         self.pairgeneration = PairGeneration0(600)

#         self.fc_aspect_h = nn.Linear(400, 200)
#         self.fc_opinion_h = nn.Linear(400, 200)
#         self.fc_sentiment_h = nn.Linear(400, 200)
#         self.fc_pair_h = nn.Linear(800, 400)
#         self.fc_pair_sentiment_h = nn.Linear(800, 400)

#         self.fc_aspect = nn.Linear(200, 3)
#         self.fc_opinion = nn.Linear(200, 3)
#         self.fc_sentiment = nn.Linear(200, 4)
#         self.fc_pair = nn.Linear(400, 3)
#         self.fc_pair_sentiment = nn.Linear(400, 4)

#         self.fc_s0 = nn.Linear(600, 200)
#         self.fc_r_s = nn.Linear(600, 200)
#         self.fc_r_s_w = nn.Linear(600, 200)

#     def forward(self, inputs, mask):
        
#         # input
#         text_indices, mask, local_adj, global_adj, global_adj1, global_adj2, relevant_sentences, relevant_sentences_presentation, m_, n_, local_adj_pmi,_,_,_ = inputs 
#         # prepare 
#         batch_size = text_indices.shape[0]
#         sentence_len = text_indices.shape[1]
#         rele_sen_num = relevant_sentences.shape[1]
#         rele_sen_len = relevant_sentences_presentation.shape[-1]
#         # process input
#         # global_adj1 = torch.reshape(global_adj1, (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
#         global_adj1 = torch.reshape(global_adj1.permute(0,2,1,3), (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
#         global_adj2 = torch.reshape(global_adj2.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
#         # process global adj to get formal adj and norm 
#         # graph 0
#         formal_global_adj = generate_formal_adj(global_adj)
#         norm_global_adj = preprocess_adj(formal_global_adj)
#         # graph 1
#         formal_global_adj1 = generate_formal_adj(global_adj1)
#         norm_global_adj1 = preprocess_adj(formal_global_adj1)
#         # graph 2
#         formal_global_adj2 = generate_formal_adj(global_adj2)
#         norm_global_adj2 = preprocess_adj(formal_global_adj2)
#         # get sentence mask
#         mask_ = mask.view(-1,1)
#         '''get the features of sentence0'''
#         # input sentnece s_0
#         text_len = torch.sum(text_indices != 0, dim=-1)
#         word_embeddings = self.embed(text_indices)
#         text = self.text_embed_dropout(word_embeddings)
#         text_out, (_, _) = self.lstm(text, text_len) # 32, 13, 600
#         text_out_narrow = self.fc_s0(text_out)
#         '''get the features of sentence1 to sentence_k'''
#         # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
#         relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
#         sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
#         sentence_embedding = self.embed(relevant_sentences_presentation)     
#         sentence_text_ = self.text_embed_dropout(sentence_embedding)
#         sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
#         ones = torch.ones_like(sentence_text_len)
#         # sentence word features
#         sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len))
#         sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
#         sentence_text_out_narrow = self.fc_r_s_w(sentence_text_out)
#         # sentence features
#         sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
#         sentence_text_out1_narrow = self.fc_r_s(sentence_text_out1)
#         '''graph 2'''
#         # process formal features to match the formal adj; first row then column
#         # global graph: row -> relevant sentence feature, column -> relevant sentence word feature
#         relevant_sentence_word_features = torch.reshape(sentence_text_out_narrow, (batch_size, rele_sen_num*rele_sen_len, -1))
#         sentence0_sentence_word_features = text_out_narrow
#         formal_global_features2 = torch.cat([sentence0_sentence_word_features, relevant_sentence_word_features], 1)
#         # GCN with local global graph1 to get relevant sentences features
#         global_text_out2 = self.gcn2(norm_global_adj2, formal_global_features2)[:, :sentence_len, :]
#         '''graph 1'''
#         # process formal features to match the formal adj; first row then column
#         # global graph: row -> relevant sentence feature, column -> relevant sentence word feature
#         relevant_sentence_features = sentence_text_out1_narrow
#         relevant_sentence_word_features = torch.reshape(sentence_text_out_narrow, (batch_size, rele_sen_num*rele_sen_len, -1))
#         formal_global_features1 = torch.cat([relevant_sentence_features, relevant_sentence_word_features], 1)
#         # GCN with global graph1 to get relevant sentences features
#         global_text_out1 = self.gcn1(norm_global_adj1, formal_global_features1)[:, :rele_sen_num, :]
#         '''graph 0'''
#         # process formal features to match the formal adj
#         # pdb.set_trace()
#         formal_global_features = torch.cat([text_out_narrow, global_text_out1], 1)
#         # GCN with local global graph
#         global_text_out = self.gcn0(norm_global_adj, formal_global_features)[:, :sentence_len, :]
#         # pdb.set_trace()
#         '''get features for final classification'''
#         # unified features
#         unified_text = torch.cat([text_out_narrow.float(), global_text_out.float(),  global_text_out2.float()], -1)
#         # pair generation
#         pair_text = self.pairgeneration(unified_text)
#         # AE and OE scores
#         aspect_probs = self.fc_aspect(self.fc_aspect_h(unified_text.float())).contiguous().view(-1, 3)
#         opinion_probs = self.fc_opinion(self.fc_opinion_h(unified_text.float())).contiguous().view(-1, 3)
#         sentiment_probs = self.fc_sentiment(self.fc_sentiment_h(unified_text.float())).contiguous().view(-1, 4)

#         # pair mask
#         pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
#                      + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,text_out.shape[1])
#         pair_mask_ = pair_mask.view(-1,1)
#         pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])

#         # pair scores     
#         pair_probs = self.fc_pair(self.fc_pair_h(pair_text.float()*pair_mask_grid)).contiguous().view(-1, 3)

#         # pair sentiment scores     
#         pair_sentiment_probs = self.fc_pair_sentiment(self.fc_pair_sentiment_h(pair_text.float()*pair_mask_grid)).contiguous().view(-1, 4)

#         return aspect_probs, opinion_probs, sentiment_probs, pair_probs, pair_sentiment_probs


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

def pairgeneration(text):
    hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
    hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
    output = hidden_1 + hidden_2
    return output

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
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
        output = self.gcn_layer1(F)
        return output

class TS3(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TS3, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.text_embed_dropout = nn.Dropout(0.5)
        self.pairgeneration = PairGeneration0(900)

        self.gcn_local = GCNforFeature_1(600, 300, 0.5)
        self.gcn0 = GCNforFeature_1(600, 300, 0.5)
        self.gcn1 = GCNforFeature_1(600, 300, 0.5)
        self.gcn2 = GCNforFeature_1(600, 300, 0.5)

        self.gcn3 = GCNforFeature_2(600, 300, 150, 0.5)
        self.gcn4 = GCNforFeature_2(600, 300, 150, 0.5)
        self.gcn5 = GCNforFeature_2(600, 300, 150, 0.5)

        self.aspect_opinion_classifier = SequenceLabelForAO(2400, 3, 0.5)
        
        self.triple_classifier = SequenceLabelForTriple(2400, 3, 0.5)
    def forward(self, inputs, mask):
        
        # input
        text_indices, mask, global_adj, global_adj1, global_adj2, relevant_sentences, relevant_sentences_presentation, m_, n_, _,_,_ = inputs 
        # prepare 
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        rele_sen_num = relevant_sentences.shape[1]
        rele_sen_len = relevant_sentences_presentation.shape[-1]
        # process input
        # global_adj1 = torch.reshape(global_adj1.permute(0,2,1,3), (batch_size, rele_sen_num, rele_sen_num*rele_sen_len))
        # global_adj2 = torch.reshape(global_adj2.permute(0,2,1,3), (batch_size, sentence_len, rele_sen_num*rele_sen_len))
        # process global adj to get formal adj and norm 
        # graph 0
        formal_global_adj = generate_formal_adj(global_adj)
        norm_global_adj = preprocess_adj(formal_global_adj)
        # graph 1
        # formal_global_adj1 = generate_formal_adj(global_adj1)
        # norm_global_adj1 = preprocess_adj(formal_global_adj1)
        # graph 2
        # formal_global_adj2 = generate_formal_adj(global_adj2)
        # norm_global_adj2 = preprocess_adj(formal_global_adj2)
        # get sentence mask
        mask_ = mask.view(-1,1)
        '''get the features of sentence0'''
        # input sentnece s_0
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        text = self.text_embed_dropout(word_embeddings)
        text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
        '''get the features of sentence1 to sentence_k'''
        # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
        relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
        sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
        sentence_embedding = self.embed(relevant_sentences_presentation_)     
        
        ones = torch.ones_like(sentence_text_len)
        # sentence word features
        sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_embedding, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
        sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
        # sentence features
        sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
        # get local and global adj
        attention_feature_local, attention_feature_global = text_out, sentence_text_out1
        attention_feature_global0 = torch.reshape(sentence_text_out, (batch_size, rele_sen_num*rele_sen_len,-1))
        attention_local, attention_global = \
                                        torch.matmul(attention_feature_local, attention_feature_local.permute(0, 2, 1)), \
                                        torch.matmul(attention_feature_local, attention_feature_global.permute(0, 2, 1))
        attention_global0 = torch.matmul(attention_feature_local, attention_feature_global0.permute(0, 2, 1))
        attention_local, attention_global = F.softmax(attention_local, -1), F.softmax(attention_global, -1)
        attention_global0 = F.softmax(attention_global0, -1)
        formal_attention_local, formal_attention_global = generate_formal_adj(attention_local), generate_formal_adj(attention_global)
        formal_attention_global0 = generate_formal_adj(attention_global0)
        norm_local_adj, norm_global_adj = preprocess_adj(formal_attention_local), preprocess_adj( formal_attention_global)
        norm_global_adj0 = preprocess_adj( formal_attention_global0)
        # get features
        formal_global_features = torch.cat([text_out, sentence_text_out1], 1)
        formal_local_features = torch.cat([text_out, text_out], 1)
        formal_global_features0 = torch.cat([text_out, torch.reshape(sentence_text_out, (batch_size, rele_sen_num*rele_sen_len,-1))], 1)
        # gcn
        if self.opt.gcn_layers_in_graph0 == 1:
            local_text_out = self.gcn0(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
            global_text_out0 = self.gcn2(norm_global_adj0, formal_global_features0)[:, :sentence_len, :]
        elif self.opt.gcn_layers_in_graph0 == 2:
            local_text_out = self.gcn3(norm_local_adj, formal_local_features)[:, :sentence_len, :]
            global_text_out = self.gcn4(norm_global_adj, formal_global_features)[:, :sentence_len, :]
            global_text_out0 = self.gcn5(norm_global_adj0, formal_global_features0)[:, :sentence_len, :]
        # unified features
        unified_text = torch.cat([text_out.float(), local_text_out.float(), global_text_out0.float()], -1)
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