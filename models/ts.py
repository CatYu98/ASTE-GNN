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
        output = torch.matmul(adj.cuda(), hidden.cuda()) / denom.cuda()
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

class PairGeneration0(nn.Module):
    def __init__(self, in_dim):
        super(PairGeneration0, self).__init__() # 32,13,300   32,300,13
        # self.weight1 = nn.Parameter(torch.FloatTensor(in_dim, int(in_dim/2)))
        # self.weight2 = nn.Parameter(torch.FloatTensor(in_dim, int(in_dim/2)))
        self.weight1 = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
        self.bias1 = nn.Parameter(torch.FloatTensor(in_dim))
    def forward(self, text):
        hidden_1 = torch.unsqueeze(text,1).repeat(1,text.shape[1],1,1)
        hidden_2 = torch.unsqueeze(text,2).repeat(1,1,text.shape[1],1)
        # hidden_1 = torch.matmul(hidden_1, self.weight1) 
        # hidden_1 = hidden_1 + self.bias1
        # hidden_2 = torch.matmul(hidden_2, self.weight2)
        output = torch.cat((hidden_1, hidden_2),-1)
        # hidden_1 = torch.matmul(hidden_1, self.weight1)
        # hidden_2 = torch.matmul(hidden_2, self.weight2)
        # output = hidden_1 + hidden_2

        return output

class GridGeneration(nn.Module):
    def __init__(self):
        super(GridGeneration, self).__init__() # 32,13,300   32,300,13
        # self.weight1 = nn.Parameter(torch.FloatTensor(in_dim, int(in_dim/2)))
        # self.weight2 = nn.Parameter(torch.FloatTensor(in_dim, int(in_dim/2)))
        # self.weight1 = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
        # self.weight2 = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
        # self.bias1 = nn.Parameter(torch.FloatTensor(in_dim))
    def forward(self, aspect_span_embd, opinion_span_embd):
        hidden_1 = torch.unsqueeze(aspect_span_embd,1).repeat(1,aspect_span_embd.shape[1],1,1)
        hidden_2 = torch.unsqueeze(opinion_span_embd,2).repeat(1,1,opinion_span_embd.shape[1],1)
        # hidden_1 = torch.matmul(hidden_1, self.weight1) 
        # hidden_1 = hidden_1 + self.bias1
        # hidden_2 = torch.matmul(hidden_2, self.weight2)
        output = torch.cat((hidden_1, hidden_2),-1)
        # hidden_1 = torch.matmul(hidden_1, self.weight1)
        # hidden_2 = torch.matmul(hidden_2, self.weight2)
        # output = hidden_1 + hidden_2

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

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X.float())
        F = torch.mm(A, X)
        F = self.gcn_layer1(F)
        
        F = self.dropout(F)
        F = torch.mm(A, F)
        output = self.gcn_layer2(F)
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

class TS(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TS, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        
        self.aspect_opinion_classifier = SequenceLabelForAO(1200, 1, 0.5)
        self.pair_sentiment_classifier =  MultiNonLinearClassifier(1200, 4, 0.5)
        self.triple_classifier = SequenceLabelForTriple(1200, 3, 0.5)
        self.pair_fc = nn.Linear(1200, 600)
        self.triple_fc = nn.Linear(1200, 600)
        self.pair_cls = nn.Linear(600, 3)
        self.triple_cls = nn.Linear(600, 4)
        self.sentiment_cls = nn.Linear(2400, 4)
        
        self.text_embed_dropout = nn.Dropout(0.5)
        self.pairgeneration = PairGeneration0(600)
        self.gridgeneration = GridGeneration()

        self.gcn = GCNforFeature_2(600, 300, 150, 0.5)
    def forward(self, inputs, mask):
        # input
        text_indices, mask, train_sentences_indices = inputs
        
        # prepare 
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        top_k_num = int(sentence_len*0.5)
        rele_sen_num = 3
        rele_sen_len = train_sentences_indices.shape[2]
        # get sentence mask
        mask_ = mask.view(-1,1)
        
        # input sentnece s_0
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        text = self.text_embed_dropout(word_embeddings)
        # pdb.set_trace()
        text_out, (text_out1, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
        '''text_out1 = torch.cat([text_out1[0], text_out1[0]], dim=-1)'''
        # flat input all sentence and get embeddings according to glove 
        '''train_sentences_indices_ = torch.reshape(train_sentences_indices, (-1, train_sentences_indices.shape[-1]))
        train_sentence_text_len = torch.sum(train_sentences_indices_!=0, dim=-1)
        train_sentences_embeddings = self.embed(train_sentences_indices_)'''
        # dropout and reshape embedding to the init shape 
        '''train_sentences_embeddings_ = self.text_embed_dropout(train_sentences_embeddings)
        train_sentences_text = torch.reshape(train_sentences_embeddings_, (-1, train_sentences_embeddings_.shape[-2], train_sentences_embeddings_.shape[-1]))'''
        # lstm to encode train sentences
        '''ones = torch.ones_like(train_sentence_text_len)
        train_sentences_text_out, (train_sentences_text_out1, b_) = self.lstm_(train_sentences_text, torch.where(train_sentence_text_len <= 0, ones, train_sentence_text_len).cpu())
        train_sentences_text_out = torch.reshape(train_sentences_text_out, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], train_sentences_indices.shape[2], train_sentences_text_out.shape[-1]))
        train_sentences_text_out1 = torch.reshape(train_sentences_text_out1, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], -1))'''
        
        # using retrieved sentence to get the text_out
        # compute the similarity between the target sentence and all the train sentences [batch_size, sentence_len, all_train_sentences_num]
        '''all_similarity = F.softmax(torch.matmul(text_out1.unsqueeze(1), train_sentences_text_out1.permute(0,2,1)), dim=-1).squeeze(1)'''
        # select the top M sentences for the target sentence
        '''selected = torch.topk(all_similarity, rele_sen_num, -1, sorted=True, largest=True)
        selected_train_sentences_text_out1 = torch.gather(train_sentences_text_out1, 1 , selected.indices.unsqueeze(-1).repeat(1, 1, train_sentences_text_out1.shape[-1]))
        selected_train_sentences_text_out = torch.gather(train_sentences_text_out, 1 , selected.indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, train_sentences_text_out.shape[-2], train_sentences_text_out.shape[-1]))'''
        # construct graphs!!!!!!!!!!!!!!
        '''relevant_sentence_features = selected_train_sentences_text_out1
        relevant_sentence_word_features = torch.reshape(selected_train_sentences_text_out, (batch_size, rele_sen_num*rele_sen_len, -1))
        attention_feature_local, attention_feature_global0 = text_out, relevant_sentence_word_features
        attention_global0 = F.softmax(torch.matmul(attention_feature_local, attention_feature_global0.permute(0, 2, 1)), -1)
        formal_attention_global0 = generate_formal_adj(attention_global0)
        norm_global_adj0 = preprocess_adj( formal_attention_global0)
        formal_global_features0 = torch.cat([text_out, relevant_sentence_word_features], 1)'''
        # gcn 
        '''global_text_out0 = self.gcn(norm_global_adj0, formal_global_features0)[:, :sentence_len, :]
        unified_text = torch.cat([text_out.float(), global_text_out0.float()], -1)'''
        # pair generation
        pair_text = self.pairgeneration(text_out)
        '''pair_text = self.pairgeneration(unified_text)'''
        # AE and OE scores (BIO tagging)
        aspect_probs, opinion_probs = self.aspect_opinion_classifier(pair_text.float())
        '''ones_probs = torch.ones_like(aspect_probs)
        aspect_probs = torch.cat([ones_probs-aspect_probs, aspect_probs], -1)
        opinion_probs = torch.cat([ones_probs-opinion_probs, opinion_probs], -1)'''
        # aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
        # select the candidate aspect terms and opinion terms
        # the index of maximum probs in aspect terms and opinion terms [batch_size, sentence_len, sentence_len]
        # aspect_index, opinion_index = aspect_probs.argmax(dim=-1), opinion_probs.argmax(dim=-1)
        # the maximum probs of aspect terms and opinion terms [batch_size, sentence_len, sentence_len]
        # aspect_likelihood = torch.gather(aspect_probs, -1, aspect_index.unsqueeze(-1)).squeeze(-1)
        # opinion_likelihood = torch.gather(opinion_probs, -1, opinion_index.unsqueeze(-1)).squeeze(-1)
        likelihood_mask_up_tri = torch.tensor(np.triu(np.ones((sentence_len, sentence_len)), k=0)).unsqueeze(0).repeat(batch_size, 1, 1)
        # aspect_likelihood = F.softmax(aspect_probs, dim=-1)[:, :, :, 1] * likelihood_mask_up_tri.cuda()
        # opinion_likelihood = F.softmax(opinion_probs, dim=-1)[:, :, :, 1] * likelihood_mask_up_tri.cuda()
        aspect_likelihood = F.softmax(aspect_probs.squeeze(-1) * likelihood_mask_up_tri.cuda(), dim=-1) 
        opinion_likelihood = F.softmax(opinion_probs.squeeze(-1) * likelihood_mask_up_tri.cuda(), dim=-1)
        # select topk aspect spans and opinion spans
        aspect_likelihood_k  = torch.topk(aspect_likelihood, 1, dim=-1, sorted=True, largest=True) 
        opinion_likelihood_k = torch.topk(opinion_likelihood, 1, dim=-1, sorted=True, largest=True)
        # select indices in the 2-th dimension
        aspect_likelihood_k_k = torch.topk(F.softmax(aspect_likelihood_k.values.squeeze(-1), dim=-1), top_k_num, dim=-1, sorted=True, largest=True)
        opinion_likelihood_k_k = torch.topk(F.softmax(opinion_likelihood_k.values.squeeze(-1), dim=-1), top_k_num, dim=-1, sorted=True, largest=True)
        # select indices in the 1-th dimension
        aspect_indices_k_shrink = torch.gather(aspect_likelihood_k.indices.squeeze(-1), 1, aspect_likelihood_k_k.indices.squeeze(-1)).unsqueeze(-1)
        opinion_indices_k_shrink = torch.gather(opinion_likelihood_k.indices.squeeze(-1), 1, opinion_likelihood_k_k.indices.squeeze(-1)).unsqueeze(-1)
        # combine indices in 2 dimensions
        aspect_indices = torch.cat([aspect_likelihood_k_k.indices.unsqueeze(-1), aspect_indices_k_shrink], dim=-1)
        opinion_indices = torch.cat([opinion_likelihood_k_k.indices.unsqueeze(-1), opinion_indices_k_shrink], dim=-1)
        # select the aspect and opinion span representation respectively
        # select aspect start and end embedding from the init sentence embedding 
        aspect_start_representation = torch.gather(text_out, 1, aspect_indices[:,:,0].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
        aspect_end_representation = torch.gather(text_out, 1, aspect_indices[:,:,1].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
        opinion_start_representation = torch.gather(text_out, 1, opinion_indices[:,:,0].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
        opinion_end_representation = torch.gather(text_out, 1, opinion_indices[:,:,1].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
        # aspect_start_representation = torch.gather(unified_text, 1, aspect_indices[:,:,0].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
        # aspect_end_representation = torch.gather(unified_text, 1, aspect_indices[:,:,1].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
        # opinion_start_representation = torch.gather(unified_text, 1, opinion_indices[:,:,0].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
        # opinion_end_representation = torch.gather(unified_text, 1, opinion_indices[:,:,1].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
        # concat the start and end embedding to get the span embedding
        aspect_span_representation = torch.cat([aspect_start_representation, aspect_end_representation], dim=-1)
        opinion_span_representation = torch.cat([opinion_start_representation, opinion_end_representation], dim=-1)
        # concat aspect and opinion span embedding to get the grid representations
        grid_representation = self.gridgeneration(aspect_span_representation, opinion_span_representation)
        # sentiment prediction
        grid_sentiment_probs = self.sentiment_cls(grid_representation)

        return aspect_probs, opinion_probs, aspect_indices, opinion_indices, grid_sentiment_probs

# class TS(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(TS, self).__init__()
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
#         self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        
#         self.aspect_opinion_classifier = SequenceLabelForAO(1500, 2, 0.5)
#         self.pair_sentiment_classifier =  MultiNonLinearClassifier(1200, 4, 0.5)
#         self.triple_classifier = SequenceLabelForTriple(1200, 3, 0.5)
#         self.pair_fc = nn.Linear(1200, 600)
#         self.triple_fc = nn.Linear(1200, 600)
#         self.pair_cls = nn.Linear(600, 3)
#         self.triple_cls = nn.Linear(600, 4)
#         self.sentiment_cls = nn.Linear(3000, 4)
        
#         self.text_embed_dropout = nn.Dropout(0.5)
#         self.pairgeneration = PairGeneration0(600)
#         self.gridgeneration = GridGeneration()

#         self.gcn = GCNforFeature_2(600, 300, 150, 0.5)
#     def forward(self, inputs, mask):
#         # input
#         text_indices, mask, train_sentences_indices = inputs
        
#         # prepare 
#         batch_size = text_indices.shape[0]
#         sentence_len = text_indices.shape[1]
#         top_k_num = 3 #int(sentence_len*0.34)
#         rele_sen_num = 3
#         rele_sen_len = train_sentences_indices.shape[2]
#         # get sentence mask
#         mask_ = mask.view(-1,1)
        
#         # input sentnece s_0
#         text_len = torch.sum(text_indices != 0, dim=-1)
#         word_embeddings = self.embed(text_indices)
#         text = self.text_embed_dropout(word_embeddings)
#         text_out, (text_out1, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
#         text_out1 = torch.cat([text_out1[0], text_out1[0]], dim=-1)
#         # flat input all sentence and get embeddings according to glove 
#         train_sentences_indices_ = torch.reshape(train_sentences_indices, (-1, train_sentences_indices.shape[-1]))
#         train_sentence_text_len = torch.sum(train_sentences_indices_!=0, dim=-1)
#         train_sentences_embeddings = self.embed(train_sentences_indices_)
#         # dropout and reshape embedding to the init shape 
#         train_sentences_embeddings_ = self.text_embed_dropout(train_sentences_embeddings)
#         train_sentences_text = torch.reshape(train_sentences_embeddings_, (-1, train_sentences_embeddings_.shape[-2], train_sentences_embeddings_.shape[-1]))
#         # lstm to encode train sentences
#         ones = torch.ones_like(train_sentence_text_len)
#         train_sentences_text_out, (train_sentences_text_out1, b_) = self.lstm_(train_sentences_text, torch.where(train_sentence_text_len <= 0, ones, train_sentence_text_len).cpu())
#         train_sentences_text_out = torch.reshape(train_sentences_text_out, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], train_sentences_indices.shape[2], train_sentences_text_out.shape[-1]))
#         train_sentences_text_out1 = torch.reshape(train_sentences_text_out1, (train_sentences_indices.shape[0], train_sentences_indices.shape[1], -1))
        
#         # using retrieved sentence to get the text_out
#         # compute the similarity between the target sentence and all the train sentences [batch_size, sentence_len, all_train_sentences_num]
#         all_similarity = F.softmax(torch.matmul(text_out1.unsqueeze(1), train_sentences_text_out1.permute(0,2,1)), dim=-1).squeeze(1)
#         # select the top M sentences for the target sentence
#         selected = torch.topk(all_similarity, rele_sen_num, -1, sorted=True, largest=True)
#         selected_train_sentences_text_out1 = torch.gather(train_sentences_text_out1, 1 , selected.indices.unsqueeze(-1).repeat(1, 1, train_sentences_text_out1.shape[-1]))
#         selected_train_sentences_text_out = torch.gather(train_sentences_text_out, 1 , selected.indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, train_sentences_text_out.shape[-2], train_sentences_text_out.shape[-1]))
#         # construct graphs!!!!!!!!!!!!!!
#         relevant_sentence_features = selected_train_sentences_text_out1
#         relevant_sentence_word_features = torch.reshape(selected_train_sentences_text_out, (batch_size, rele_sen_num*rele_sen_len, -1))
#         attention_feature_local, attention_feature_global0 = text_out, relevant_sentence_word_features
#         attention_global0 = F.softmax(torch.matmul(attention_feature_local, attention_feature_global0.permute(0, 2, 1)), -1)
#         formal_attention_global0 = generate_formal_adj(attention_global0)
#         norm_global_adj0 = preprocess_adj( formal_attention_global0)
#         formal_global_features0 = torch.cat([text_out, relevant_sentence_word_features], 1)
#         # gcn 
#         global_text_out0 = self.gcn(norm_global_adj0, formal_global_features0)[:, :sentence_len, :]
#         unified_text = torch.cat([text_out.float(), global_text_out0.float()], -1)
#         # pair generation
#         # pair_text = self.pairgeneration(text_out)
#         pair_text = self.pairgeneration(unified_text)
#         # AE and OE scores (BIO tagging)
#         aspect_probs, opinion_probs = self.aspect_opinion_classifier(pair_text.float())
#         # aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
#         # select the candidate aspect terms and opinion terms
#         # the index of maximum probs in aspect terms and opinion terms [batch_size, sentence_len, sentence_len]
#         # aspect_index, opinion_index = aspect_probs.argmax(dim=-1), opinion_probs.argmax(dim=-1)
#         # the maximum probs of aspect terms and opinion terms [batch_size, sentence_len, sentence_len]
#         # aspect_likelihood = torch.gather(aspect_probs, -1, aspect_index.unsqueeze(-1)).squeeze(-1)
#         # opinion_likelihood = torch.gather(opinion_probs, -1, opinion_index.unsqueeze(-1)).squeeze(-1)
#         likelihood_mask_up_tri = torch.tensor(np.triu(np.ones((sentence_len, sentence_len)), k=0)).unsqueeze(0).repeat(batch_size, 1, 1)
#         aspect_likelihood = F.softmax(aspect_probs, dim=-1)[:, :, :, 1] * likelihood_mask_up_tri.cuda()
#         opinion_likelihood = F.softmax(opinion_probs, dim=-1)[:, :, :, 1] * likelihood_mask_up_tri.cuda()
#         # select topk aspect spans and opinion spans
#         aspect_likelihood_k  = torch.topk(aspect_likelihood, 1, dim=-1, sorted=True, largest=True) 
#         opinion_likelihood_k = torch.topk(opinion_likelihood, 1, dim=-1, sorted=True, largest=True)
#         # select indices in the 2-th dimension
#         aspect_likelihood_k_k = torch.topk(aspect_likelihood_k.values, top_k_num, dim=-2, sorted=True, largest=True)
#         opinion_likelihood_k_k = torch.topk(opinion_likelihood_k.values, top_k_num, dim=-2, sorted=True, largest=True)
#         # select indices in the 1-th dimension
#         aspect_indices_k_shrink = torch.gather(aspect_likelihood_k.indices.squeeze(-1), 1, aspect_likelihood_k_k.indices.squeeze(-1)).unsqueeze(-1)
#         opinion_indices_k_shrink = torch.gather(opinion_likelihood_k.indices.squeeze(-1), 1, opinion_likelihood_k_k.indices.squeeze(-1)).unsqueeze(-1)
#         # combine indices in 2 dimensions
#         aspect_indices = torch.cat([aspect_likelihood_k_k.indices, aspect_indices_k_shrink], dim=-1)
#         opinion_indices = torch.cat([opinion_likelihood_k_k.indices, opinion_indices_k_shrink], dim=-1)
#         # select the aspect and opinion span representation respectively
#         # select aspect start and end embedding from the init sentence embedding 
#         # aspect_start_representation = torch.gather(text_out, 1, aspect_indices[:,:,0].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
#         # aspect_end_representation = torch.gather(text_out, 1, aspect_indices[:,:,1].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
#         # opinion_start_representation = torch.gather(text_out, 1, opinion_indices[:,:,0].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
#         # opinion_end_representation = torch.gather(text_out, 1, opinion_indices[:,:,1].unsqueeze(-1).repeat(1,1,text_out.shape[-1]))
#         aspect_start_representation = torch.gather(unified_text, 1, aspect_indices[:,:,0].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
#         aspect_end_representation = torch.gather(unified_text, 1, aspect_indices[:,:,1].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
#         opinion_start_representation = torch.gather(unified_text, 1, opinion_indices[:,:,0].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
#         opinion_end_representation = torch.gather(unified_text, 1, opinion_indices[:,:,1].unsqueeze(-1).repeat(1,1,unified_text.shape[-1]))
#         # concat the start and end embedding to get the span embedding
#         aspect_span_representation = torch.cat([aspect_start_representation, aspect_end_representation], dim=-1)
#         opinion_span_representation = torch.cat([opinion_start_representation, opinion_end_representation], dim=-1)
#         # concat aspect and opinion span embedding to get the grid representations
#         grid_representation = self.gridgeneration(aspect_span_representation, opinion_span_representation)
#         # sentiment prediction
#         grid_sentiment_probs = self.sentiment_cls(grid_representation)
#         # aspect_span_representation = torch.gather(pair_text, 1, aspect_indices.unsqueeze(-1))
#         # opinion_span_representation = torch.gather(pair_text, 1, opinion_indices.unsqueeze(-1))
#         # import pdb; pdb.set_trace()
#         # pair scores    
#         # pair_probs_, triple_probs_ = self.triple_classifier(pair_text.float())
#         # !!!
#         # pair_hidden = self.pair_fc(pair_text)
#         # triple_hidden = self.triple_fc(pair_text)
#         # triple_atten = F.softmax(torch.matmul(pair_hidden, triple_hidden.permute(0,1,3,2)), dim=-1)
#         # pair_atten = F.softmax(torch.matmul(triple_hidden, pair_hidden.permute(0,1,3,2)), dim=-1)
#         # pair_hidden = torch.matmul(pair_atten, pair_hidden)
#         # triple_hidden = torch.matmul(triple_atten, triple_hidden)
#         # pair_probs_, triple_probs_ = self.pair_cls(pair_hidden), self.triple_cls(triple_hidden)
#         # !!!
#         # pair_probs = pair_probs_.contiguous().view(-1, 3)
#         # triple_probs = triple_probs_.contiguous().view(-1, 4)

#         return aspect_probs, opinion_probs, aspect_indices, opinion_indices, grid_sentiment_probs