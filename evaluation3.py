import numpy as np 
import pdb

def convert_to_list(y_aspect, y_opinion, y_sentiment):
    y_aspect_list = []
    y_opinion_list = []
    y_sentiment_list = []
    
    for seq_aspect, seq_opinion, seq_sentiment in zip(y_aspect, y_opinion, y_sentiment):
        l_a = []
        l_o = []
        l_s = []
        for label_dist_a, label_dist_o, label_dist_s in zip(seq_aspect, seq_opinion, seq_sentiment):
            l_a.append(np.argmax(label_dist_a))
            l_o.append(np.argmax(label_dist_o))
            # if not np.any(label_dist_s):
            #     l_s.append(0)
            # else:
            l_s.append(np.argmax(label_dist_s))
        y_aspect_list.append(l_a)
        y_opinion_list.append(l_o)
        y_sentiment_list.append(l_s)
    return y_aspect_list, y_opinion_list, y_sentiment_list

def score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, mask):

    begin = 1
    inside = 2

    # predicted sentiment distribution for aspect terms that are correctly extracted
    pred_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # gold sentiment distribution for aspect terms that are correctly extracted
    rel_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # sentiment distribution for terms that get both span and sentiment predicted correctly
    correct_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # sentiment distribution in original data
    total_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}

    polarity_map = {1: 'pos', 2: 'neg', 3: 'neu', 4:'con'}

    # count of predicted conflict aspect term
    predicted_conf = 0

    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1
                # if not train_op:
                if true_sentiment[i][num]!=0:
                    total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1 # aspect extraction correct
                        # if not train_op:
                            # do not count conflict examples
                        if true_sentiment[i][num]!=0:
                            rel_count[polarity_map[true_sentiment[i][num]]]+=1 # real sentiment when aspect is correct
                            if predict_sentiment[i][num]!=0:
                                pred_count[polarity_map[predict_sentiment[i][num]]]+=1 # predict sentiment when aspect is correct
                            if true_sentiment[i][num] == predict_sentiment[i][num]:
                                correct_count[polarity_map[true_sentiment[i][num]]]+=1 # aspect and sentiment are correct

                        else:
                            predicted_conf += 1 # aspect is correct but sentiment is none



        for pred in predict:
            if pred == begin:
                predicted += 1 # aspect nums predicted; relavent is aspect nums real

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    # print(f_aspect)

    acc_s, f_s, f_absa = 0, 0, 0

    # # if not train_op:
    # num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
    # num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
    # num_total = total_count['pos']+total_count['neg']+total_count['neu']

    # acc_s = num_correct_overall/(num_correct_aspect+1e-6)
    
    # p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
    # r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
    
    # p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
    # r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

    # p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
    # r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

    # pr_s = (p_pos+p_neg+p_neu)/3,0
    # re_s = (r_pos+r_neg+r_neu)/3,0
    # # F1 score for AS only
    # print(pr_s, re_s)
    # if pr_s+re_s != 0:
    #     f_s = 2*pr_s*re_s/(pr_s+re_s)
    # else:
    #     f_s = 0
    
    # precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
    # recall_absa = num_correct_overall/(num_total+1e-6)
    # # F1 score of the end-to-end task
    # f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)

    return f_aspect, acc_s, f_s, f_absa

def get_metric(y_true_aspect, y_predict_aspect, y_true_opinion, y_predict_opinion, y_true_sentiment, y_predict_sentiment, mask):
    f_a, f_o = 0, 0
    true_aspect, true_opinion, true_sentiment = y_true_aspect, y_true_opinion, y_true_sentiment
    predict_aspect, predict_opinion, predict_sentiment = convert_to_list(y_predict_aspect, y_predict_opinion, y_predict_sentiment)
    # predict_aspect, predict_sentiment = y_predict_aspect, y_predict_sentiment
    f_aspect, acc_s, f_s, f_absa = score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, mask)
    f_opinion, _, _, _ = score(true_opinion, predict_opinion, true_sentiment, true_sentiment, 0)

    return f_aspect, f_opinion, acc_s, f_s, f_absa

def score3(true_aspect, predict_aspect, true_opinion, predict_opinion, true_sentiment, predict_sentiment, mask, train_op=0):

    begin = 1
    inside = 2

    # predicted sentiment distribution for aspect terms that are correctly extracted
    pred_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # gold sentiment distribution for aspect terms that are correctly extracted
    rel_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # sentiment distribution for terms that get both span and sentiment predicted correctly
    correct_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}
    # sentiment distribution in original data
    total_count = {'pos':0, 'neg':0, 'neu':0, 'con':0}

    polarity_map = {1: 'pos', 2: 'neg', 3: 'neu', 4:'con'}

    # count of predicted conflict aspect term
    predicted_conf = 0

    correct_a, predicted_a, relevant_a = 0, 0, 0
    correct_o, predicted_o, relevant_o = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq_aspect = true_aspect[i]
        predict_aspect = predict_aspect[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1
                if not train_op:
                    if true_sentiment[i][num]!=0:
                        total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1 # aspect extraction correct
                        if not train_op:
                            # do not count conflict examples
                            if true_sentiment[i][num]!=0:
                                rel_count[polarity_map[true_sentiment[i][num]]]+=1 # real sentiment when aspect is correct
                                if predict_sentiment[i][num]!=0:
                                    pred_count[polarity_map[predict_sentiment[i][num]]]+=1 # predict sentiment when aspect is correct
                                if true_sentiment[i][num] == predict_sentiment[i][num]:
                                    correct_count[polarity_map[true_sentiment[i][num]]]+=1 # aspect and sentiment are correct

                            else:
                                predicted_conf += 1 # aspect is correct but sentiment is none



        for pred in predict:
            if pred == begin:
                predicted += 1 # aspect nums predicted; relavent is aspect nums real

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)



    acc_s, f_s, f_absa = 0, 0, 0

    if not train_op:
        num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
        num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
        num_total = total_count['pos']+total_count['neg']+total_count['neu']

        acc_s = num_correct_overall/(num_correct_aspect+1e-6)
       
        p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
        r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
        
        p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
        r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

        p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
        r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

        pr_s = (p_pos+p_neg+p_neu)/3,0
        re_s = (r_pos+r_neg+r_neu)/3,0
        # F1 score for AS only
        if pr_s+re_s != 0:
            f_s = 2*pr_s*re_s/(pr_s+re_s)
        else:
            f_s = 0
       
        precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
        recall_absa = num_correct_overall/(num_total+1e-6)
        # F1 score of the end-to-end task
        f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)

    return f_aspect, acc_s, f_s, f_absa

def find_pair_(label_matrix):
    length = len(label_matrix)
    triple = []
    for i in range(length):
        for j in range(length):
            aspect_index, opinion_index = [], []
            if label_matrix[i][j] == 1:
                tem_a, tem_o = i, j
                # 1down 2right 3rightdown
                direction = 0
                while label_matrix[tem_a][tem_o] != 0:
                    if label_matrix[tem_a][tem_o] == 1:
                        # if direction == 3:
                        # save
                        aspect_index.append(tem_a)
                        opinion_index.append(tem_o)
                        # elif direction == 1:
                        #     flag = 1
                        #     for t in range(j, tem_o):
                        #         if label_matrix[tem_a][t]!=2:
                        #             flag=0
                        #     if flag == 1:
                        #         aspect_index.append(tem_a)
                        #     else: break
                        # elif direction == 2:
                        #     flag = 1
                        #     for t in range(i, tem_a):
                        #         if label_matrix[i][tem_o]!=2:
                        #             flag=0
                        #     if flag == 1:
                        #         opinion_index.append(tem_o)
                        #     else: break
                        # jump
                        if label_matrix[tem_a+1][tem_o]==2 and label_matrix[tem_a][tem_o+1]==2 and label_matrix[tem_a+1][tem_o+1]==2:
                            direction=3 # right down
                            tem_a+=1
                            tem_o+=1
                            continue
                        # elif label_matrix[tem_a+1][tem_o]==2 and label_matrix[tem_a][tem_o+1]==0 and label_matrix[tem_a+1][tem_o+1]==0:
                        elif (tem_a<length and tem_o<length and label_matrix[tem_a+1][tem_o]==2 and label_matrix[tem_a][tem_o+1]==0 and label_matrix[tem_a+1][tem_o+1]==0) or\
                            (tem_a<length and tem_o>length and label_matrix[tem_a+1][tem_o]==2):
                            direction=1 # down
                            tem_a+=1
                            continue
                        # elif label_matrix[tem_a+1][tem_o]==0 and label_matrix[tem_a][tem_o+1]==2 and label_matrix[tem_a+1][tem_o+1]==0:
                        elif (tem_a<length and tem_o<length and label_matrix[tem_a+1][tem_o]==0 and label_matrix[tem_a][tem_o+1]==2 and label_matrix[tem_a+1][tem_o+1]==0) or\
                            (tem_a>length and tem_o<length and label_matrix[tem_a][tem_o+1]==2):
                            direction=2 # right
                            tem_o+=1
                            continue
                        else:
                            break
                    
                    elif label_matrix[tem_a][tem_o] == 2:
                        # save
                        if direction == 3:
                            aspect_index.append(tem_a)
                            opinion_index.append(tem_o)
                        elif direction == 1:
                            flag = 1
                            for t in range(j, tem_o):
                                if label_matrix[tem_a][t]!=2:
                                    flag=0
                            if flag == 1:
                                aspect_index.append(tem_a)
                            else: break
                        elif direction == 2:
                            flag = 1
                            for t in range(i, tem_a):
                                if label_matrix[i][tem_o]!=2:
                                    flag=0
                            if flag == 1:
                                opinion_index.append(tem_o)
                            else: break
                        # jump
                        if tem_a<length and tem_o<length and label_matrix[tem_a+1][tem_o]==2 and label_matrix[tem_a][tem_o+1]==2 and label_matrix[tem_a+1][tem_o+1]==2:
                            direction=3
                            tem_a+=1
                            tem_o+=1
                            continue
                        elif (tem_a<length and tem_o<length and label_matrix[tem_a+1][tem_o]==2 and label_matrix[tem_a][tem_o+1]==0 and label_matrix[tem_a+1][tem_o+1]==0) or\
                            (tem_a<length and tem_o>length and label_matrix[tem_a+1][tem_o]==2):
                            direction=1
                            tem_a+=1
                            continue
                        elif (tem_a<length and tem_o<length and label_matrix[tem_a+1][tem_o]==0 and label_matrix[tem_a][tem_o+1]==2 and label_matrix[tem_a+1][tem_o+1]==0) or\
                            (tem_a>length and tem_o<length and label_matrix[tem_a][tem_o+1]==2):
                            direction=2
                            tem_o+=1
                            continue
                        else:break

            if aspect_index != [] and opinion_index != []:
                triple.append([aspect_index, opinion_index])
    return triple

def find_pair(label_matrix):
    length = len(label_matrix)
    triple = []
    for i in range(length):
        for j in range(length):
            aspect_index, opinion_index = [], []
            # import pdb; pdb.set_trace()
            if label_matrix[i][j] == 1:
                # aspect_index.append(i)
                # opinion_index.append(j)
                col , row, tem_len = j, i, 1
                save_length = []
                while True:
                    while col+1 < len(label_matrix[1]) and label_matrix[row][col+1] == 2:
                        col += 1 
                        tem_len += 1
                    save_length.append(tem_len)
                    tem_len = 1
                    if row+1 < len(label_matrix) and label_matrix[row+1][j] == 2:
                        row += 1
                        col = j
                    else: break
                max_len = max(save_length)
                aspect_index = [idx for idx in range(i, row+1)]
                opinion_index = [idx for idx in range(j, j + max_len)]

            if aspect_index != [] and opinion_index != []:
                triple.append([aspect_index, opinion_index])
    return triple

def find_grid_term(label_matrix):
    length = len(label_matrix)
    triple = []
    for i in range(length):
        for j in range(length):
            aspect_index, opinion_index = [], []
            if label_matrix[i][j] == 1:
                aspect_index.append(i)
                opinion_index.append(j)
                tem_a, tem_o = i, j
                for t in range(tem_a, length):
                    if label_matrix[t][tem_o]==2:
                        aspect_index.append(t)
                for t in range(tem_o, length):
                    if label_matrix[tem_a][t]==2:
                        opinion_index.append(t)

            if aspect_index != [] and opinion_index != [] and aspect_index == opinion_index:
                triple.append(aspect_index)
    return triple

def find_term_span(label_matrix):
    length = len(label_matrix)
    spans = []
    for i in range(length):
        for j in range(length):
            if label_matrix[i][j] == 1:
                if i <= j :spans.append([i,j])
                # else: spans.append([j, i])
    return spans

def find_pair_sentiment(label_matrix, sentiment_label_matrix):
    length = len(label_matrix)
    triple = []
    for i in range(length):
        for j in range(length):
            aspect_index, opinion_index = [], []
            # import pdb; pdb.set_trace()
            if label_matrix[i][j] == 1:
                # aspect_index.append(i)
                # opinion_index.append(j)
                col , row, tem_len = j, i, 1
                save_length = []
                while True:
                    while col+1 < len(label_matrix[1]) and label_matrix[row][col+1] == 2:
                        col += 1 
                        tem_len += 1
                    save_length.append(tem_len)
                    tem_len = 1
                    if row+1 < len(label_matrix) and label_matrix[row+1][j] == 2:
                        row += 1
                        col = j
                    else: break
                max_len = max(save_length)
                aspect_index = [idx for idx in range(i, row+1)]
                opinion_index = [idx for idx in range(j, j + max_len)]

            if aspect_index != [] and opinion_index != [] and sentiment_label_matrix[i][j] != 0:
                triple.append([aspect_index, opinion_index, sentiment_label_matrix[i][j]])
    return triple

def find_term(label_sequence):
    term = []
    for i in range(len(label_sequence)):
        tem_term = []
        if label_sequence[i] == 1:
            tem_term.append(i)
            for j in range(i+1, len(label_sequence)):
                if label_sequence[j] == 2:
                    tem_term.append(j)
                else: break
        else:continue
        if tem_term != []:
            term.append(tem_term)
    return term

def compute_sentiment(true_aspect, predict_aspect, true_sentiment, predict_sentiment):
    begin = 1
    inside = 2
    # predicted sentiment distribution for aspect terms that are correctly extracted
    pred_count = {'pos':0, 'neg':0, 'neu':0}
        # gold sentiment distribution for aspect terms that are correctly extracted
    rel_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution for terms that get both span and sentiment predicted correctly
    correct_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution in original data
    total_count = {'pos':0, 'neg':0, 'neu':0}

    polarity_map = {1: 'pos', 2: 'neg', 3: 'neu', 0:'null'}

    # count of predicted conflict aspect term
    predicted_conf = 0

    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1
                if true_sentiment[i][num]!=0:
                    total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1
                        # do not count conflict examples
                        if true_sentiment[i][num]!=0:
                            rel_count[polarity_map[true_sentiment[i][num]]]+=1
                            if predict_sentiment[i][num] != 0:
                                pred_count[polarity_map[predict_sentiment[i][num]]]+=1
                            if true_sentiment[i][num] == predict_sentiment[i][num]:
                                correct_count[polarity_map[true_sentiment[i][num]]]+=1

                        else:
                            predicted_conf += 1



        for pred in predict:
            if pred == begin:
                predicted += 1

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    acc_s, f_s, f_absa = 0, 0, 0

    num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
    num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
    num_total = total_count['pos']+total_count['neg']+total_count['neu']

    acc_s = num_correct_overall/(num_correct_aspect+1e-6)
    
    p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
    r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
    
    p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
    r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

    p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
    r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

    pr_s = (p_pos+p_neg+p_neu)/3.0
    re_s = (r_pos+r_neg+r_neu)/3.0
    # F1 score for AS only
    f_s = 2*pr_s*re_s/(pr_s+re_s+1e-6)
    
    precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
    recall_absa = num_correct_overall/(num_total+1e-6)
    # F1 score of the end-to-end task
    f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)
    return f_s, f_absa

def find_pair_according_to_ao_span(label_matrix, aspect_span, opinion_span):
    '''
    aspect_list is a list containing all aspect terms, which are found by the find_term function
    opinion_list is a list similar to the aspect_list
    label_matrix is the grid label for pair
    '''
    pairs = []
    for aspect in aspect_span:
        for opinion in opinion_span:
            a_start, a_end = aspect
            o_start, o_end = opinion
            flag = 1
            if a_start == a_end and o_start == o_end:
                flag = 0
            if aspect[0]<=opinion[0]:
                if flag == 0 and label_matrix[aspect[0]][opinion[0]] == 1:
                    pairs.append([aspect, opinion])
                if flag == 1 and label_matrix[aspect[0]][opinion[0]] == 2:
                    pairs.append([aspect, opinion])
                # if flag == 1:
                #     if aspect[-1]<=opinion[-1] and label_matrix[aspect[-1]][opinion[-1]] == 2:pairs.append([aspect, opinion])
                #     elif aspect[-1]>opinion[-1] and label_matrix[opinion[-1]][aspect[-1]] == 2:pairs.append([aspect, opinion])
                # else:
                    # pairs.append([aspect, opinion])
            if aspect[0]>opinion[0]:
                if flag == 0 and label_matrix[opinion[0]][aspect[0]] == 1:
                    pairs.append([aspect, opinion])
                if flag == 1 and label_matrix[opinion[0]][aspect[0]] == 2:
                    pairs.append([aspect, opinion])
                # if flag == 1:
                #     if aspect[-1]<=opinion[-1] and label_matrix[aspect[-1]][opinion[-1]] == 2:pairs.append([aspect, opinion])
                #     elif aspect[-1]>opinion[-1] and label_matrix[opinion[-1]][aspect[-1]] == 2:pairs.append([aspect, opinion])
                # else:
                #     pairs.append([aspect, opinion])
                # pairs.append([aspect, opinion])
    return pairs

def find_triple_sentiment_according_to_ao_span(label_matrix, aspect_list, opinion_list):
    '''
    aspect_list is a list containing all aspect terms, which are found by the find_term function
    opinion_list is a list similar to the aspect_list
    label_matrix is the grid label for triple
    '''
    triples = []
    for aspect in aspect_list:
        for opinion in opinion_list:
            if aspect[0]<opinion[0] and label_matrix[aspect[0]][opinion[0]] != 0:
                sentiment = label_matrix[aspect[0]][opinion[0]]
                triples.append([aspect, opinion, sentiment])
            if aspect[0]>opinion[0] and label_matrix[opinion[0]][aspect[0]] != 0:
                sentiment = label_matrix[opinion[0]][aspect[0]]
                triples.append([aspect, opinion, sentiment])
    return triples

def find_triple_sentiment_according_to_pair_span(label_matrix, pair_list):
    triples = []
    for pair in pair_list:
        aspect, opinion = pair
        if aspect[0]<=opinion[0] and label_matrix[aspect[0]][opinion[0]] != 0:
            sentiment = label_matrix[aspect[0]][opinion[0]]
            triples.append([aspect, opinion, sentiment])
        if aspect[0]>opinion[0] and label_matrix[opinion[0]][aspect[0]] != 0:
            sentiment = label_matrix[opinion[0]][aspect[0]]
            triples.append([aspect, opinion, sentiment])
    return triples

def find_pair_according_to_ao_gts(label_matrix, aspect_list, opinion_list):
    '''
    aspect_list is a list containing all aspect terms, which are found by the find_term function
    opinion_list is a list similar to the aspect_list
    label_matrix is the grid label for pair
    '''
    pairs = []
    for aspect in aspect_list:
        for opinion in opinion_list:
            count = [0, 0, 0]
            for i in aspect:
                for j in opinion:
                    if aspect[0] > opinion[0]: count[int(label_matrix[j][i])] += 1
                    else: count[int(label_matrix[i][j])] += 1
            if count[2]+1 == int((len(aspect) * len(opinion))):
                if aspect[0] > opinion[0]:
                    if label_matrix[opinion[0]][aspect[0]] == 1:
                        pairs.append([aspect, opinion])
                else:
                    if label_matrix[aspect[0]][opinion[0]] == 1:
                        pairs.append([aspect, opinion])
    return pairs

def find_triple_sentiment_according_to_pair_gts(label_matrix, pair_list):
    triples = []
    for pair in pair_list:
        aspect_list, opinion_list = pair
        count = [0, 0, 0, 0]
        for i in aspect_list:
            for j in opinion_list:
                if i > j: count[int(label_matrix[j][i])] += 1
                else: count[int(label_matrix[i][j])] += 1
        if sum(count[1:]) == 0:
            continue
        if count[3] >= count[2] and count[3] >= count[1]:
            sentiment = 3
        elif count[2] >= count[1] and count[2] >= count[3]:
            sentiment = 2
        elif count[1] >= count[3] and count[1] >= count[2]:
            sentiment = 1
        else: continue
        triples.append([aspect_list, opinion_list, sentiment])
    return triples

def find_triple_sentiment_according_to_ao_gts(label_matrix, aspect_list, opinion_list):
    '''
    aspect_list is a list containing all aspect terms, which are found by the find_term function
    opinion_list is a list similar to the aspect_list
    label_matrix is the grid label for triple
    '''
    triples = []
    for aspect in aspect_list:
        for opinion in opinion_list:
            count = [0, 0, 0, 0]
            for i in aspect:
                for j in opinion:
                    if i<j: count[int(label_matrix[i][j])] += 1
                    else: count[int(label_matrix[j][i])] += 1
            if sum(count[1:]) == 0:
                continue
            if count[3] >= count[2] and count[3] >= count[1] and count[3] >= int((len(aspect) * len(opinion)) / 2):
                sentiment = 3
            elif count[2] >= count[1] and count[2] >= count[3] and count[2] >= int((len(aspect) * len(opinion)) / 2):
                sentiment = 2
            elif count[1] >= count[3] and count[1] >= count[2] and count[1] >= int((len(aspect) * len(opinion)) / 2):
                sentiment = 1
            else: continue
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                input()
            triples.append([aspect, opinion, sentiment])
    return triples

if __name__ == '__main__':
    # test
    tem1 = [[0,1,2,0],[0,2,2,0],[0,2,2,0],[0,0,0,0]]
    tem1_ = [[0,2,2,0],[0,2,2,0],[0,2,2,0],[0,0,0,0]]
    tem2 = [[0,1,2,0],[0,2,2,0],[0,0,0,0],[0,0,0,0]]
    tem2_ = [[0,3,3,0],[0,3,3,0],[0,0,0,0],[0,0,0,0]]
    tem3 = [[0,1,2,2],[0,2,2,2],[0,0,0,0],[0,0,0,0]]
    tem3_ = [[0,2,2,2],[0,2,2,2],[0,0,0,0],[0,0,0,0]]
    tem4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 1, 2, 2, 0],\
            [0, 0, 0, 0, 0, 2, 2, 2, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    tem4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 2, 2, 2, 0],\
            [0, 0, 0, 0, 0, 2, 2, 2, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    tem5 = [[0,1,2,2],[0,0,0,0],[1,0,0,0],[0,0,0,0]]
    print(find_pair(tem1))
    print(find_pair(tem2))
    print(find_pair(tem3))
    print(find_pair(tem4))
    print(find_pair(tem5))
    # import torch
    # print(torch.tensor(tem3).shape)
    pdb.set_trace()
    tem = [0,1,2,2,0,0,1,0,0,1,2]
    print(find_term(tem))