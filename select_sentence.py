import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
import pdb
from tqdm import tqdm
import time
import heapq
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

# datasets = ['res14', 'lap14', 'res15', 'res16']
# dataset = sys.argv[1]

# if dataset not in datasets:
# 	sys.exit("wrong dataset name")

# w_s_tf_idf = load_dict('../'+dataset+'_tf_idf')

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def read_all_sentence(domain):
    '''read all sentence (train/dev/test) to get the representation of relevant sentences'''
    train_data = open('./ASTE-Data-V2/'+domain+'/train_triplets.txt','r').readlines()
    dev_data = open('./ASTE-Data-V2/'+domain+'/dev_triplets.txt','r').readlines()
    test_data = open('./ASTE-Data-V2/'+domain+'/test_triplets.txt','r').readlines()

    train_sentences = [line.split('####')[0] for line in train_data]
    dev_sentences = [line.split('####')[0] for line in dev_data]
    test_sentences = [line.split('####')[0] for line in test_data]
    all_sentences = train_sentences + dev_sentences + test_sentences
    return all_sentences

def read_text(fnames):
    '''a string: sentence1\nsentence2\n...sentencen\n'''
    text = ''
    for fname in fnames:
        fin = open(fname)
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines)):
            text += lines[i].split('####')[0].lower().strip()+'\n'
    return text

def read_triplets(fnames):
    '''a list: [[([2], [5], 'NEG')], [(),()], [], ..., []]'''
    triplets = []
    for fname in fnames:
        fin = open(fname) 
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines)):
            triple = eval(lines[i].split('####')[1])
            triplets.append(triple)
    return triplets

def select_sentence(domain):
    # read data
    all_sentences = read_all_sentence(domain)
    # read num of sentences in train, dev, test
    train_len, dev_len, test_len = \
        len(open('./ASTE-Data-V2/'+domain+'/train_triplets.txt').readlines()), len(open('./ASTE-Data-V2/'+domain+'/dev_triplets.txt').readlines()), len(open('./ASTE-Data-V2/'+domain+'/test_triplets.txt').readlines())
    print(train_len, dev_len, test_len)
    # load bert model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # get sentence representation using BERT 
    print('get sentence representation using BERT!')
    all_sentences_feature = []
    for sentence in tqdm(all_sentences):
        tokens = tokenizer.tokenize(sentence)
        input = tokenizer(sentence, max_length=len(tokens)+2, truncation=True, padding='max_length', return_tensors='pt')
        features = bert_model(input.input_ids, input.attention_mask)[0][0][0]
        all_sentences_feature.append(features)
    # select relevant sentences
    print('select relevant sentences!')
    relevant_sentences_index = []
    relevant_sentences_val = []
    for sen_idx in tqdm(range(len(all_sentences))):
        tem_cos_sim = []
        sentence = all_sentences[sen_idx]
        sentence_feature = all_sentences_feature[sen_idx]
        tem_r_sentence_idx = []
        for feature_idx in range(len(all_sentences_feature)):
            feature = all_sentences_feature[feature_idx]
            if sen_idx == feature_idx:
                tem_cos_sim.append(0)
            else:
                tem_cos_sim.append(cos(sentence_feature, feature))
            # if cos(sentence_feature, feature) > gamma:
            #     # if feature_idx != sen_idx:
            #     tem_r_sentence_idx.append(feature_idx)
        tem_cos_sim = torch.tensor(tem_cos_sim)
        tem_r_sentence_val, tem_r_sentence_idx = tem_cos_sim.cuda().topk(1, dim=0, largest=True, sorted=True)
        # tem_r_sentence_idx = heapq.nlargest(len(all_sentences[sen_idx].strip().split()), range(len(tem_cos_sim)), tem_cos_sim.__getitem__)
        relevant_sentences_index.append(tem_r_sentence_idx)
        relevant_sentences_val.append(tem_r_sentence_val)

    train_relevant_sentences_index = relevant_sentences_index[:train_len]
    dev_relevant_sentences_index = relevant_sentences_index[train_len:train_len + dev_len]
    test_relevant_sentences_index = relevant_sentences_index[train_len + dev_len:]

    train_relevant_sentences_val = relevant_sentences_val[:train_len]
    dev_relevant_sentences_val = relevant_sentences_val[train_len:train_len + dev_len]
    test_relevant_sentences_val = relevant_sentences_val[train_len + dev_len:]

    # write relevant sentences, 3 denotes the top 3 
    train_r_sentences = open('./ASTE-Rele-Sentences/'+domain+'/train_r1.txt','a')
    dev_r_sentences = open('./ASTE-Rele-Sentences/'+domain+'/dev_r1.txt','a')
    test_r_sentences = open('./ASTE-Rele-Sentences/'+domain+'/test_r1.txt','a')

    for i in train_relevant_sentences_index:
        for ii in i :
            train_r_sentences.write(str(ii.cpu().numpy().tolist())+' ')
        train_r_sentences.write('\n')

    for i in dev_relevant_sentences_index:
        for ii in i :
            dev_r_sentences.write(str(ii.cpu().numpy().tolist())+' ')
        dev_r_sentences.write('\n')

    for i in test_relevant_sentences_index:
        for ii in i :
            test_r_sentences.write(str(ii.cpu().numpy().tolist())+' ')
        test_r_sentences.write('\n')

    train_r_sentences.close()
    dev_r_sentences.close()
    test_r_sentences.close()

if __name__ == '__main__':
    select_sentence('res14')
    select_sentence('res15')
    select_sentence('lap14')
    select_sentence('res16')