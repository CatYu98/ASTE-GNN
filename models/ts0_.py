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
        output = self.gcn_layer1(F)
        return output

class TS0_(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TS0_, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.lstm_ = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'LSTM')
        self.text_embed_dropout = nn.Dropout(0.5)
        self.pairgeneration = PairGeneration0(150)

        self.gcn0 = GCNforFeature_1(600, 300, 0.5)
        # self.gcn1 = GCNforFeature_1(600, 300, 0.5)
        # self.gcn2 = GCNforFeature_2(600, 300, 150, 0.5)
        self.gcn1 = GCNforFeature_1(300, 150, 0.5)
        self.gcn2 = GCNforFeature_2(600, 300, 150, 0.5)

        self.fc_aspect = nn.Linear(600, 3)
        self.fc_opinion = nn.Linear(600, 3)
        self.fc_sentiment = nn.Linear(600, 4)
        self.fc_pair = nn.Linear(1200, 3)
        self.fc_pair_sentiment = nn.Linear(1200, 4)

        self.features1 = nn.Linear(600, 300)
        self.features2 = nn.Linear(600, 300)

        self.aspect_classifier = MultiNonLinearClassifier(300, 3, 0.5)
        self.opinion_classifier = MultiNonLinearClassifier(300, 3, 0.5)
        self.aspect_opinion_classifier = SequenceLabelForAO(150, 3, 0.5)
        self.customize_aspect_opinion_classifier = CustomizeSequenceLabelForAO(300, 3, 0.5)
        self.aspect_opinion_sentiment_classifier = SequenceLabelForAOS(150, 3, 0.5)

        self.sentiment_classifier = MultiNonLinearClassifier(150, 4, 0.5)

        self.pair_classifier = MultiNonLinearClassifier(300, 3, 0.5)
        self.pair_sentiment_classifier =  MultiNonLinearClassifier(300, 4, 0.5)
        
        self.triple_classifier = SequenceLabelForTriple(300, 3, 0.5)
        self.grid_classifier = SequenceLabelForGrid(300, 3, 0.5)

        self.norm1 = nn.LayerNorm(3)
        self.norm2 = nn.LayerNorm(4)

    def forward(self, inputs, mask):
        
        # input
        text_indices, mask, local_adj, global_adj, relevant_sentences, relevant_sentences_presentation, m_, n_, local_adj_pmi,_,_,_ = inputs
        # prepare 
        batch_size = text_indices.shape[0]
        sentence_len = text_indices.shape[1]
        rele_sen_num = relevant_sentences.shape[1]
        rele_sen_len = relevant_sentences_presentation.shape[-1]
        # process local adj to get formal adj
        norm_local_adj = preprocess_adj(local_adj).float()
        # process global adj to get formal adj and norm 
        formal_global_adj = generate_formal_adj(global_adj)
        norm_global_adj = preprocess_adj(formal_global_adj)
        # get sentence mask
        mask_ = mask.view(-1,1)
        # input sentnece s_0
        text_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        text = self.text_embed_dropout(word_embeddings)
        text_out, (_, _) = self.lstm(text, text_len.cpu()) # 32, 13, 600
        # use local graph
        local_text_out = self.gcn0(norm_local_adj, text_out)
        # pdb.set_trace()
        # relevant sentences, for every sentence s_0, there are T relevant sentences s_1, s_2, ..., s_T
        relevant_sentences_presentation_ = torch.reshape(relevant_sentences_presentation, (-1, relevant_sentences_presentation.shape[-1]))
        sentence_text_len = torch.sum(relevant_sentences_presentation_!= 0, dim=-1)
        sentence_embedding = self.embed(relevant_sentences_presentation)     
        sentence_text_ = self.text_embed_dropout(sentence_embedding)
        sentence_text = torch.reshape(sentence_text_, (-1, sentence_text_.shape[-2], sentence_text_.shape[-1]))
        
        ones = torch.ones_like(sentence_text_len)
        sentence_text_out, (sentence_text_out1, b_) = self.lstm_(sentence_text, torch.where(sentence_text_len <= 0, ones, sentence_text_len).cpu())
        sentence_text_out = torch.reshape(sentence_text_out, (relevant_sentences.shape[0], relevant_sentences.shape[1], sentence_text_out.shape[-2], sentence_text_out.shape[-1]))
        sentence_text_out1 = torch.reshape(sentence_text_out1, (relevant_sentences.shape[0], relevant_sentences.shape[1], -1))
        # process formal features to match the formal adj
        text_out_ = self.features1(text_out)
        sentence_text_out1_ = self.features1(sentence_text_out1)
        formal_global_features = torch.cat([text_out_, sentence_text_out1_], 1)
        # GCN with local global graph
        if self.opt.gcn_layers_in_graph0 == 1:
            global_text_out = self.gcn1(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        elif self.opt.gcn_layers_in_graph0 == 2:
            global_text_out = self.gcn2(norm_global_adj, formal_global_features)[:, :sentence_len, :]
        # unified features
        # unified_text = torch.cat([text_out.float(), global_text_out.float(), local_text_out.float()], -1)
        # unified_text = torch.cat([text_out.float(), global_text_out.float()], -1)
        unified_text = global_text_out
        # unified_text = global_text_out.float()
        # pair generation
        pair_text = self.pairgeneration(unified_text)
        # AE and OE scores
        if self.opt.emb_for_ao == 'private_single':
            aspect_probs = self.fc_aspect(unified_text.float()).contiguous().view(-1, 3)
            opinion_probs = self.fc_opinion(unified_text.float()).contiguous().view(-1, 3)
            sentiment_probs = self.fc_sentiment(unified_text.float()).contiguous().view(-1, 4)
        elif self.opt.emb_for_ao == 'private_multi':
            aspect_probs = self.aspect_classifier(unified_text.float()).contiguous().view(-1, 3)
            opinion_probs = self.opinion_classifier(unified_text.float()).contiguous().view(-1, 3)
            sentiment_probs = self.sentiment_classifier(unified_text.float()).contiguous().view(-1, 4)
        elif self.opt.emb_for_ao == 'pair_shared_multi':
            aspect_probs, opinion_probs = self.aspect_opinion_classifier(unified_text.float())
            aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
            sentiment_probs = self.sentiment_classifier(unified_text.float()).contiguous().view(-1, 4)
        elif self.opt.emb_for_ao == 'triple_shared_multi':
            aspect_probs, opinion_probs, sentiment_probs = self.aspect_opinion_sentiment_classifier(unified_text.float())
            aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
            sentiment_probs = sentiment_probs.contiguous().view(-1, 4)
        elif self.opt.emb_for_ao == 'pair_customize_shared_multi':
            aspect_probs, opinion_probs = self.customize_aspect_opinion_classifier(unified_text.float())
            aspect_probs, opinion_probs = aspect_probs.contiguous().view(-1, 3), opinion_probs.contiguous().view(-1, 3)
            sentiment_probs = self.sentiment_classifier(unified_text.float()).contiguous().view(-1, 4)
        '''# pair mask
        tem_ones, tem_zeros = torch.ones_like(aspect_probs.argmax(-1)), torch.zeros_like(aspect_probs.argmax(-1))
        aspect_mask = torch.where(aspect_probs.argmax(-1)>0, tem_ones, tem_zeros)
        opinion_mask = torch.where(opinion_probs.argmax(-1)>0, tem_ones, tem_zeros)
        pair_mask = torch.unsqueeze(aspect_mask.view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
                     + torch.unsqueeze(opinion_mask.view(text_out.shape[0],-1),2).repeat(1,1,text_out.shape[1])
        # pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
        #              + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1),2).repeat(1,1,text_out.shape[1])
        # pair_mask_grid_ = torch.where(pair_mask>0, tem_ones, tem_zeros)
        # pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
        pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1, pair_text.shape[-1])'''
        # pair mask
        # pair_mask = torch.unsqueeze(F.normalize((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1), p=2, dim=-1),1).repeat(1,text_out.shape[1],1)\
        #              + torch.unsqueeze(F.normalize((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1), p=2, dim=-1),2).repeat(1,1,text_out.shape[1])
        # pair_mask_ = pair_mask.view(-1,1)
        # pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
        # normalization?

        # pair mask for pair prediction (according to aspect and opinion probs)
        pair_mask = torch.unsqueeze((aspect_probs[:,-1]+aspect_probs[:,-2]).view(text_out.shape[0],-1),1).repeat(1,text_out.shape[1],1)\
                     + torch.unsqueeze((opinion_probs[:,-1]+opinion_probs[:,-2]).view(text_out.shape[0],-1), 2).repeat(1,1,text_out.shape[1])
        pair_mask_ = pair_mask.view(-1,1)
        pair_mask_grid = torch.unsqueeze(pair_mask,-1).repeat(1,1,1,pair_text.shape[-1])
        # pair scores     
        # if self.opt.emb_for_ps == 'private_single':
        #     pair_probs_ = self.fc_pair(pair_text.float()*pair_mask_grid)
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = self.self.fc_sentiment(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)
        # elif self.opt.emb_for_ps == 'private_multi':
        #     pair_probs_ = self.pair_classifier(pair_text.float()*pair_mask_grid)
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = self.pair_sentiment_classifier(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)
        # elif self.opt.emb_for_ps == 'shared_multi':
        #     pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float()*pair_mask_grid)
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)
        # I love YGX!
        if self.opt.emb_for_ps == 'private_single':
            pair_probs_ = self.fc_pair(pair_text.float())
            pair_probs = pair_probs_.contiguous().view(-1, 3)
            pair_sentiment_probs = self.self.fc_sentiment(pair_text.float()).contiguous().view(-1, 4)
        elif self.opt.emb_for_ps == 'private_multi':
            pair_probs_ = self.pair_classifier(pair_text.float())
            pair_probs = pair_probs_.contiguous().view(-1, 3)
            pair_sentiment_probs = self.pair_sentiment_classifier(pair_text.float()).contiguous().view(-1, 4)
        elif self.opt.emb_for_ps == 'shared_multi':
            pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float())
            pair_probs = pair_probs_.contiguous().view(-1, 3)
            pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)

        aspect_grid_label, opinion_grid_label = self.grid_classifier(pair_text.float())
        aspect_grid_label = aspect_grid_label.contiguous().view(-1, 3)
        opinion_grid_label = opinion_grid_label.contiguous().view(-1, 3)

        
        # if self.opt.emb_for_ps == 'private_single':
        #     pair_probs_ = self.fc_pair(pair_text.float())
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = self.self.fc_sentiment(pair_text.float()).contiguous().view(-1, 4)
        # elif self.opt.emb_for_ps == 'private_multi':
        #     pair_probs_ = self.pair_classifier(pair_text.float())
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = self.pair_sentiment_classifier(pair_text.float()).contiguous().view(-1, 4)
        # elif self.opt.emb_for_ps == 'shared_multi':
        #     pair_probs_, pair_sentiment_probs_ = self.triple_classifier(pair_text.float())
        #     pair_probs = pair_probs_.contiguous().view(-1, 3)
        #     pair_sentiment_probs = pair_sentiment_probs_.contiguous().view(-1, 4)

        # pdb.set_trace()
        # pair sentiment mask using pair probs
        # pair_sentiment_mask = torch.unsqueeze(pair_probs_[:, :, :, 1] + pair_probs_[:, :, :, 1], -1).repeat(1, 1, 1, pair_text.shape[-1])
        # pair sentiment prediction using sentiment probs
        # pair_sentiment_probs_s = \
        #     (((torch.unsqueeze(sentiment_probs.view(batch_size, sentence_len, -1), 1).repeat(1, sentence_len, 1, 1) \
        #         + torch.unsqueeze(sentiment_probs.view(batch_size, sentence_len, -1), 2).repeat(1, 1, sentence_len, 1)) / 2 ).view(-1, 4)) * pair_mask_
        # pair sentiment scrores
        # pair_sentiment_probs = self.fc_pair_sentiment(pair_text.float()*pair_sentiment_mask_p).contiguous().view(-1, 4)
        '''pair_sentiment_probs = self.fc_pair_sentiment(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)'''
        '''pair_sentiment_probs = self.pair_sentiment_classifier(pair_text.float()*pair_mask_grid).contiguous().view(-1, 4)'''
        # pair_sentiment_probs = (pair_sentiment_probs_s + pair_sentiment_probs)/2

        # aspect_probs = self.norm1(aspect_probs)
        # opinion_probs = self.norm1(opinion_probs)
        # sentiment_probs = self.norm2(sentiment_probs)
        # pair_probs = self.norm1(pair_probs)
        # pair_sentiment_probs = self.norm2(pair_sentiment_probs)

        return F.log_softmax(aspect_probs, dim=-1), F.log_softmax(opinion_probs, dim=-1), F.log_softmax(sentiment_probs, dim=-1),\
                    F.log_softmax(aspect_grid_label, dim=-1), F.log_softmax(opinion_grid_label, dim=-1),\
                        F.log_softmax(pair_probs, dim=-1), F.log_softmax(pair_sentiment_probs, dim=-1)
        # return F.softmax(aspect_probs, dim=-1), F.softmax(opinion_probs, dim=-1), F.softmax(sentiment_probs, dim=-1), F.softmax(pair_probs, dim=-1), F.softmax(pair_sentiment_probs, dim=-1)
        # , torch.softmax(pair_probs,dim=-1), torch.softmax(pair_sentiment_probs,dim=-1)