import operator
import sys
import pdb
import numpy as np

def pad(save_dic):
    max_len = max([len(x) for x in save_dic['spans']])
    save_dic['mask'] = []
    for i in range(len(save_dic['spans'])):
        tmp_len = len(save_dic['spans'][i])
        save_dic['spans'][i] += [[-1, -1]] * (max_len - tmp_len)
        save_dic['aspect_span_labels'][i] += [-1] * (max_len - tmp_len)
        save_dic['opinion_span_labels'][i] += [-1] * (max_len - tmp_len)
        save_dic['span_labels'][i] += [-1] * (max_len - tmp_len)
        save_dic['sentiment_labels'][i] += [-1] * (max_len - tmp_len)
        save_dic['mask'].append([1] * tmp_len + [0] * (max_len - tmp_len))
    return save_dic

def raw2span_labels(domain, phase, max_term_len = None, padding = False) -> dic:
    ''' 
    Input:
        domain: res14, lap14, res15, res16
        phase: train, dev, test
        max_term_len: the maximum span length, the default is the maximum aspect and opinion length, if given a value: min{default value, given value}
        padding: True or Flase, padding labels in save_dic to the max_term_len, and add a mask label
    Output: A dict
        spans: [[0, 0], [0, 1], [0, 2], ... , [7, 7], [7, 8], [8, 8]
        aspect_span_labels: [0, 0, ... , 0, 1, , ... , 0, 0] 1 denote the span is an aspect
        opinion_span_labels:  [0, 0, ... , 0, 1, , ... , 0, 0] 1 denote the span is an opinion
        span_labels:  [0, 0, ... , 0, 1, 0, 2, ... , 0, 0] 1 denote the span is an aspect, 2 denotes the span is an opinion
        sentiment_labels: [0, 0, ... , 0, 1, 0, 2, ... , 0, 0] 1 denote positive, 2 denotes negative, 3 denote neutral 
        mask: 1 or 0 
        gold_aspect_spans: [2,2]
        gold_aspect_spans: [5,5]
        gold_triplets: [([2, 2], [5, 5], 2)]
        padding with -1
    '''
    raw_file = open('../ASTE-Data-V2/' + domain + '/' + phase + '_triplets.txt').readlines()
    max_aspect_len, max_opinion_len = 0, 0
    for line in raw_file:
        sentence, triplets = line.split('####')[0], eval(line.split('####')[1])
        for tri in triplets:
            if len(tri[0]) > max_aspect_len:
                max_aspect_len = len(tri[0])
            if len(tri[1]) > max_opinion_len:
                max_opinion_len = len(tri[1])
    if max_term_len == None: 
        max_term_len = max(max_aspect_len, max_opinion_len)
    else:
        max_term_len = min(max_term_len, max(max_aspect_len, max_opinion_len))

    save_dic = {'spans':[], 'aspect_span_labels':[], 'opinion_span_labels':[], 'span_labels':[], 'sentiment_labels':[]}
    save_dic['gold_aspect_spans'], save_dic['gold_opinion_spans'], save_dic['gold_triplets'] = [], [], [] 
    sentiment_dic = {'POS':1, 'NEG':2, 'NEU':3, 'pos':1, 'neg':2, 'neu':3}
    for line in raw_file:
        sentence, triplets = line.split('####')[0], eval(line.split('####')[1])
        aspect, opinion, sentiment = [[tmp[0][0], tmp[0][-1]] for tmp in triplets], [[tmp[1][0], tmp[1][-1]] for tmp in triplets], [tmp[2] for tmp in triplets]
        sentence_len = len(sentence.split(' '))
        spans, aspect_span_label, opinion_span_label, span_label, sentiment_label = [], [], [], [], []
        save_dic['gold_aspect_spans'].append(aspect)
        save_dic['gold_opinion_spans'].append(opinion)
        save_dic['gold_triplets'].append([ ([tmp[0][0], tmp[0][-1]], [tmp[1][0], tmp[1][-1]], sentiment_dic[tmp[2]]) for tmp in triplets])
        for i in range(sentence_len):
            for j in range(max_term_len):
                if i + j >= sentence_len: break
                spans.append([i, i+j])
                if [i, i+j] in aspect: 
                    aspect_span_label.append(1) 
                    opinion_span_label.append(0)
                    span_label.append(1)
                    sentiment_label.append(sentiment[aspect.index([i, i+j])])

                elif [i, i+j] in opinion: 
                    aspect_span_label.append(0)
                    opinion_span_label.append(1)
                    span_label.append(2)
                    sentiment.append(0)
                else: 
                    aspect_span_label.append(0)
                    opinion_span_label.append(0)
                    span_label.append(0)
                    sentiment.append(0)

        assert operator.eq(len(spans), len(aspect_span_label)) and operator.eq(len(spans), len(opinion_span_label)) and \
                operator.eq(len(spans), len(span_label)), operator.eq(len(spans), len(sentiment_label))

        save_dic['spans'].append(spans)
        save_dic['aspect_span_labels'].append(aspect_span_label)
        save_dic['opinion_span_labels'].append(opinion_span_label)
        save_dic['span_labels'].append(span_label)
        save_dic['sentiment_labels'].append(sentiment_label)
    if padding == True:
        save_dic = pad(save_dic)
    return save_dic

if __name__ == '__main__':
    train_data = raw2span_labels('res14', 'train', 8, True)
    train_data_pad = raw2span_labels('res14', 'train', 8)
    dev_data = raw2span_labels('res14', 'dev', 8)
    test_data = raw2span_labels('res14', 'test', 8)
    pdb.set_trace()