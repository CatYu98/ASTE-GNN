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
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MIN_(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MIN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True,bidirectional=True, rnn_type = 'LSTM')
        # self.lstm = nn.LSTM(300, 300, num_layers=1, batch_first=True,bidirectional=True)

        self.cnn01 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.cnn02 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.cnn0 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)

        self.aspect_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.opinion_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)

        self.cnn = nn.Conv1d(in_channels=600, out_channels=300, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.cnn1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.cnn2 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.cnn3 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.gc1 = GraphConvolution(600, 300)
        self.gc2 = GraphConvolution(300, 300)
        self.fc_aspect = nn.Linear(1800, 3)
        self.fc_opinion = nn.Linear(1800, 3)
        self.fc_sentiment = nn.Linear(1200, 4)
        self.sigmoid = nn.Sigmoid()
        self.text_embed_dropout = nn.Dropout(0.3)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, inputs, mask):
        import pdb; pdb.set_trace()
        text_indices, local_adj, global_adj, mask = inputs
        x_len = torch.sum(text_indices != 0, dim=-1)
        word_embeddings = self.embed(text_indices)
        sentence_output = word_embeddings


        aspect_list = []
        opinion_list = []
        sentiment_list = []
        aspect_inputs = []
        opinion_inputs = []
        sentiment_inputs = []
        aspect_inputs.append(sentence_output)
        opinion_inputs.append(sentence_output)
        sentiment_inputs.append(sentence_output)

        sentence_output = sentence_output.permute(0,2,1)

################################ shared feature extraction ################################
        for i in range(self.opt.shared_layers):
            sentence_output = self.text_embed_dropout(sentence_output)
            if i == 0:
                sentence_output_1 = self.cnn01(sentence_output)
                sentence_output_2 = self.cnn02(sentence_output)
                sentence_output = torch.cat([sentence_output_1, sentence_output_2], dim=-2)
            else:
                # sentence_output = self.cnn0(sentence_output)
                cnn0 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
                sentence_output = cnn0(sentence_output)
            word_embeddings = torch.cat([word_embeddings, sentence_output.permute(0,2,1)], dim=-1)
        init_shared_features = sentence_output
################################ private feature extraction for ATE ################################       
        aspect_output = sentence_output
        for i in range(self.opt.aspect_layers):
            aspect_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
            # aspect_output = self.aspect_cnn(aspect_output)
            aspect_output = aspect_cnn(aspect_output)
            aspect_embedding = aspect_output
################################ private feature extraction for OTE ################################ 
        opinion_output = sentence_output
        for i in range(self.opt.opinion_layers):
            opinion_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
            # opinion_output = self.opinion_cnn(opinion_output)
            opinion_output = opinion_cnn(opinion_output)
            opinion_embedding = opinion_output
################################ Pair-attention ################################ 
        # aspect2opinion = Lambda(lambda x : tf.matmul(tf.nn.l2_normalize(x[0], -1), tf.nn.l2_normalize(x[1], -1), adjoint_b=True))([aspect_embedding, opinion_embedding])
        # aspect_att_opinion = Lambda(lambda x : softmask_2d(x, sentence_mask))(aspect2opinion)
        # aspect2opinion_embedding = Lambda(lambda x: tf.concat([x[0], tf.matmul(x[1], x[2])], -1))([aspect_embedding, aspect_att_opinion, opinion_embedding])
        aspect2opinion = torch.bmm(F.normalize(aspect_embedding).transpose(1, 2), F.normalize(opinion_embedding))
        aspect2opinion_embedding = torch.cat([aspect_embedding, torch.bmm(aspect_embedding, aspect2opinion)], dim=-2)
        # print("aspect2opinion", aspect2opinion.shape)
        # print("aspect2opinion_embedding:",aspect2opinion_embedding.shape)

        opinion2aspect = torch.bmm(F.normalize(opinion_embedding).transpose(1, 2), F.normalize(aspect_embedding))
        opinion2aspect_embedding = torch.cat([opinion_embedding, torch.bmm(opinion_embedding, opinion2aspect)], dim=-2)
        # print("opinion2aspect:", opinion2aspect.shape)
        # print("opinion2aspect_embedding:", opinion2aspect_embedding.shape)
        # opinion2aspect = Lambda(lambda x : tf.matmul(tf.nn.l2_normalize(x[0], -1), tf.nn.l2_normalize(x[1], -1), adjoint_b=True))([opinion_embedding, aspect_embedding])
        # opinion_att_aspect = Lambda(lambda x : softmask_2d(x, sentence_mask))(opinion2aspect)
        # opinion2aspect_embedding = Lambda(lambda x: tf.concat([x[0], tf.matmul(x[1], x[2])], -1))([opinion_embedding, opinion_att_aspect, aspect_embedding])
################################ private feature extraction for ASC ################################ 
        sentiment_output = sentence_output
        for i in range(self.opt.opinion_layers):
            sentiment_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True)
            # opinion_output = self.opinion_cnn(opinion_output)
            sentiment_output = sentiment_cnn(opinion_output)


################################ decoder for ATE ################################            
        aspect_output = aspect_output.permute(0,2,1)
        # print("\naspect_output:", aspect_output.shape)
        # print("word_embeddings:", word_embeddings.shape)
        # print("aspect2opinion_embedding:", aspect2opinion_embedding.shape)
        aspect_output = torch.cat([aspect_output, word_embeddings, aspect2opinion_embedding.transpose(1,2)], dim=-1)
        aspect_probs = self.fc_aspect(aspect_output.contiguous().view(-1, aspect_output.shape[-1]))
        aspect_probs = F.relu(aspect_probs)
################################ decoder for OTE ################################ 
        opinion_output = opinion_output.permute(0,2,1)
        opinion_output = torch.cat([opinion_output, word_embeddings, opinion2aspect_embedding.transpose(1,2)], dim=-1)
        opinion_probs = self.fc_opinion(opinion_output.contiguous().view(-1, opinion_output.shape[-1]))
        opinion_probs = F.relu(opinion_probs)
################################ decoder for ASC ################################ 
        sentiment_output = sentiment_output.permute(0,2,1)
        sentiment_output = torch.cat([sentiment_output, word_embeddings], dim=-1)
        sentiment_probs = self.fc_sentiment(sentiment_output.contiguous().view(-1, sentiment_output.shape[-1]))
        sentiment_probs = F.relu(sentiment_probs)

        mask = mask.unsqueeze(-1)

        aspect_probs = aspect_probs.contiguous().view(aspect_output.shape[0], -1, 3)
        aspect_probs = aspect_probs * mask

        opinion_probs = opinion_probs.contiguous().view(opinion_output.shape[0], -1, 3)
        opinion_probs = opinion_probs * mask

        return aspect_probs, opinion_probs