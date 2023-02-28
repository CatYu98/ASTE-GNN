# # -*- coding: utf-8 -*-

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
# import pdb
# import torch.nn.functional as F


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
#         output = torch.matmul(adj, hidden) / denom
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output


# class LSTM(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(LSTM, self).__init__()
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True,bidirectional=True, rnn_type = 'LSTM')
#         # self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True,bidirectional=True)

#         self.cnn01 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
#         self.cnn02 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
#         self.cnn0 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)

#         self.aspect_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)

#         self.cnn = nn.Conv1d(in_channels=600, out_channels=300, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
#         self.cnn1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
#         self.cnn2 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
#         self.cnn3 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
#         self.gc1 = GraphConvolution(600, 300)
#         self.gc2 = GraphConvolution(300, 300)
#         self.fc_aspect = nn.Linear(600, 3)
#         self.fc_opinion = nn.Linear(100, 4)
#         self.fc_polarity = nn.Linear(100, 4)
#         self.sigmoid = nn.Sigmoid()
#         self.text_embed_dropout = nn.Dropout(0.3)
#         self.soft = nn.Softmax(dim=-1)

#     def forward(self, inputs, mask):
#         text_indices, adj = inputs
#         x_len = torch.sum(text_indices != 0, dim=-1)
#         x = self.embed(text_indices)

#         '''word_embeddings = x
#         sentence_output = x
#         sentence_output = self.text_embed_dropout(sentence_output)

#         sentence_output = sentence_output.permute(0,2,1)

#         # for i in range(2):
#         #     if i == 0:
#         sentence_output_1 = self.cnn01(sentence_output)
#         # print(sentence_output_1.shape)
#         sentence_output_2 = self.cnn02(sentence_output)
#         # print(sentence_output_2.shape)
#         sentence_output = torch.cat([sentence_output_1, sentence_output_2], dim=-2)
#         # print(sentence_output.shape)
#             # else:
#         sentence_output = self.cnn0(sentence_output)
#         # print(sentence_output.shape)
        
#         # for i in range(2):
#         aspect_output = sentence_output
#         aspect_output = self.aspect_cnn(aspect_output)
#         # aspect_embedding = aspect_output
#         aspect_output = aspect_output.permute(0,2,1)
#         aspect_probs = self.fc_aspect(aspect_output.contiguous().view(-1, aspect_output.shape[-1]))
#         aspect_probs = F.relu(aspect_probs)'''



#         lstm_output, (_, _) = self.lstm(x, x_len)
#         lstm_output = self.text_embed_dropout(lstm_output)
#         # lstm_output_ = lstm_output.permute(0,2,1)

#         # h_n = self.gc1(lstm_output, adj)
#         # h_n = self.gc2(h_n, adj)
        
#         # h_n = self.cnn(lstm_output_)
#         # h_n = self.cnn1(h_n)
#         # h_n = h_n.permute(0,2,1)
#         out_aspect = self.fc_aspect(lstm_output.contiguous().view(-1, lstm_output.shape[-1])) 
#         # out_aspect = F.relu(out_aspect)
#         out_aspect = nn.functional.sigmoid(out_aspect)
#         # out_aspect = self.soft(out_aspect)
#         # out_aspect = F.normalize(out_aspect, p=1, dim=-1)

#         '''mask = mask.unsqueeze(-1)
#         # out_aspect = out_aspect.contiguous().view(h_n.shape[0], -1, 3)
#         # out_aspect = out_aspect * mask
#         aspect_probs = aspect_probs.contiguous().view(aspect_output.shape[0], -1, 3)
#         aspect_probs = aspect_probs * mask
#         return aspect_probs'''
#         mask = mask.unsqueeze(-1)
#         out_aspect = out_aspect.contiguous().view(lstm_output.shape[0], -1, 3)
#         out_aspect = out_aspect * mask
#         return out_aspect

# class Highway(nn.Module):
#     """
#     Highway Network.
#     """

#     def __init__(self, size, num_layers=1, dropout=0.5):
#         """
#         :param size: size of linear layer (matches input size)
#         :param num_layers: number of transform and gate layers
#         :param dropout: dropout
#         """
#         super(Highway, self).__init__()
#         self.size = size
#         self.num_layers = num_layers
#         self.transform = nn.ModuleList()  # list of transform layers
#         self.gate = nn.ModuleList()  # list of gate layers
#         self.dropout = nn.Dropout(p=dropout)

#         for i in range(num_layers):
#             transform = nn.Linear(size, size)
#             gate = nn.Linear(size, size)
#             self.transform.append(transform)
#             self.gate.append(gate)

#     def forward(self, x):
#         """
#         Forward propagation.
#         :param x: input tensor
#         :return: output tensor, with same dimensions as input tensor
#         """
#         transformed = nn.functional.relu(self.transform[0](x))  # transform input
#         g = nn.functional.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

#         out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

#         # If there are additional layers
#         for i in range(1, self.num_layers):
#             out = self.dropout(out)
#             transformed = nn.functional.relu(self.transform[i](out))
#             g = nn.functional.sigmoid(self.gate[i](out))

#             out = g * transformed + (1 - g) * out

#         return out





# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(300, 3)

    def forward(self, inputs, mask):
        text_indices = inputs[0]
        x = self.embed(text_indices)
        x_len = torch.sum(text_indices != 0, dim=-1)
        h_n_, (h_n, _) = self.lstm(x, x_len)
        out = self.fc(h_n_.contiguous().view(-1, h_n_.shape[-1]))
        out = out.contiguous().view(out.shape[0], -1, 3)
        return out
