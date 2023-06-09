import json
import os
import scipy 
from sklearn.metrics import mean_squared_error
import math
import string
import numpy as np
import math
import matplotlib.pyplot as plt

def remove_pun(input_string):
    input_string = "".join([char for char in input_string if char not in string.punctuation])
    return input_string

def print_result(pred, gt):
    mse = mean_squared_error(pred, gt)
    corr, _ = scipy.stats.pearsonr(pred, gt)
    spearman, _ = scipy.stats.spearmanr(pred, gt)
    #print('mse:', mse)
    print('corr:', round(corr,4))
    #print('srcc:', round(spearman,4))


f = open('/Users/yuwen/Desktop/TTS/dataset/speechocean762/resource/scores.json') # path to speechocean score json
data = json.load(f)

test_file = open('/Users/yuwen/Desktop/TTS/dataset/speechocean762/test/wav.scp','r').read().splitlines() # path to speechocean test list
test_data = {}

for line in test_file:
    wavidx = line.split('\t')[0]
    test_data[wavidx] = data[wavidx]
    
def get_prediction(path):
    prediction = open(path,'r').read().splitlines()

    result_word = {}
    result_uttr = {}
    for sample in prediction:
        
        parts = sample.split(';')
        wavidx = parts[0].replace('.wav','')
        try:
            accuracy = float(parts[1].split(':')[1])
            fluency = float(parts[2].split(':')[1])
            prosodic = float(parts[3].split(':')[1])
            total = float(parts[4].split(':')[1])
        except:
            continue
        if math.isnan(accuracy):
            print(wavidx)
            continue
        valid = parts[5].split(':')[1]
        if valid=='F':
            continue
            
        asr_s = eval(parts[6].split(':')[1])
        asr_s = ' '.join([sublist[0] for sublist in asr_s])  

        gt_sen = []
        for word in test_data[wavidx]['words']:
            the_word = word['text'].lower()
            gt_sen.append(the_word)

        gt_sen = ' '.join(gt_sen)
        
        if gt_sen==asr_s:
            w_a = eval(parts[8].split(':')[1])
            w_s = eval(parts[9].split(':')[1])
            w_t = eval(parts[10].split(':')[1])
            if isinstance(w_a , float):
                w_a = [w_a]
                w_s = [w_s]
                w_t = [w_t]
            w_a = [10 if x > 10 else x for x in w_a]
            w_s = [10 if x > 10 else x for x in w_s]
            w_t = [10 if x > 10 else x for x in w_t]
            result_word[wavidx]={}
            result_word[wavidx]['word_accuracy'] = w_a
            result_word[wavidx]['word_stress'] = w_s
            result_word[wavidx]['word_total'] = w_t

        result_uttr[wavidx]={}
        result_uttr[wavidx]['accuracy'] = accuracy
        result_uttr[wavidx]['fluency'] = fluency
        result_uttr[wavidx]['prosodic'] = prosodic
        result_uttr[wavidx]['total'] = total
            
    return result_word, result_uttr 

def calculate_performance(result_word, result_uttr, wav_idx_word, wav_idx_uttr):

    gt_A = []
    gt_F = []
    gt_P = []
    gt_T = []

    pred_A = []
    pred_F = []
    pred_P = []
    pred_T = []

    for wavidx in wav_idx_uttr:
        gt_A.append(test_data[wavidx]['accuracy'])
        pred_A.append(result_uttr[wavidx]['accuracy'])
        gt_F.append(test_data[wavidx]['fluency'])
        pred_F.append(result_uttr[wavidx]['fluency'])
        gt_P.append(test_data[wavidx]['prosodic'])
        pred_P.append(result_uttr[wavidx]['prosodic'])
        gt_T.append(test_data[wavidx]['total'])
        pred_T.append(result_uttr[wavidx]['total'])
    
    print('number of utterance', len(pred_A))
    print('accuracy')
    print_result(pred_A, gt_A)
    print('fluency')
    print_result(pred_F, gt_F)
    print('prosodic')
    print_result(pred_P, gt_P)
    print('total')
    print_result(pred_T, gt_T)

    gt_w_acc = []
    gt_w_stress = []
    gt_w_total = []
    pred_w_acc = []
    pred_w_stress = []
    pred_w_total = []
    for wavidx in wav_idx_word:        
        the_gt_w_acc = []
        the_gt_w_stress = []
        the_gt_w_total = []
        the_phone_acc = []
        for word in test_data[wavidx]['words']:
            the_gt_w_acc.append(int(word['accuracy']))
            the_gt_w_stress.append(int(word['stress']))
            the_gt_w_total.append(int(word['total']))
            the_phone_acc.append(np.mean(word['phones-accuracy']))
  
        gt_w_acc.extend(the_gt_w_acc)
        gt_w_stress.extend(the_gt_w_stress)
        gt_w_total.extend(the_gt_w_total)
        pred_w_acc.extend(result_word[wavidx]['word_accuracy'])
        pred_w_stress.extend(result_word[wavidx]['word_stress'])
        pred_w_total.extend(result_word[wavidx]['word_total'])
               
    print('number of sentences for word', len(wav_idx_word))
    print('word acc')
    print_result(pred_w_acc, gt_w_acc)
    print('word stress')
    print_result(pred_w_stress, gt_w_stress)
    print('word total')
    print_result(pred_w_total, gt_w_total)



resultA_word, resultA_uttr = get_prediction('/Users/yuwen/Desktop/TTS/chatbot_code/speechocean762_test_gtb.txt')

calculate_performance(resultA_word, resultA_uttr, list(resultA_word.keys()), list(resultA_uttr.keys()))