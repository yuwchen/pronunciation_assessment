import os
import argparse
import fairseq
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import math

random.seed(1984)


class PronunciationPredictor(nn.Module):
    
    def __init__(self, ssl_model, ssl_out_dim, text_out_dim):
        super(PronunciationPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.text_out_dim = text_out_dim
        self.alignment_features = 10
        self.phone_vector = 71*2

        hidden_sen = 768*5
        hidden_word = self.ssl_features+self.text_out_dim*2+self.phone_vector+self.alignment_features

        self.output_accuracy = nn.Linear(hidden_sen, 1) 
        self.output_fluency = nn.Linear(hidden_sen, 1) 
        self.output_prosodic = nn.Linear(hidden_sen, 1) 
        self.output_total = nn.Linear(hidden_sen, 1)

        self.fusion_layer = nn.TransformerEncoderLayer(d_model=768*3, nhead=8)
        
        
        self.fusion_layer_word = nn.TransformerEncoderLayer(d_model=hidden_word, nhead=2)
        self.alignment_layer = nn.TransformerEncoderLayer(d_model=self.alignment_features, nhead=2)
        self.phonevector_layer = nn.TransformerEncoderLayer(d_model=self.phone_vector, nhead=2)

        self.word_acc = nn.Conv1d(hidden_word, 1, kernel_size=1)
        self.word_stress = nn.Conv1d(hidden_word, 1, kernel_size=1)
        self.word_total = nn.Conv1d(hidden_word, 1, kernel_size=1)

    def forward(self, wav, asr_word_embed, gt_sen_embed, gt_embed, alignment_feature, phonevector, start, end):
        res = self.ssl_model(wav, mask=False, features_only=True)
        wav_embedding_raw = res['x']

        ### word level prediction
        batch_size  = gt_embed.shape[0]
        wav_aligned = torch.zeros((gt_embed.shape[0], gt_embed.shape[1], self.ssl_features)).cuda()

        for b_idx in range(batch_size):
            for w_idx in range(len(start[b_idx])):
                start_point = int(start[b_idx][w_idx]/320)
                end_point = int(end[b_idx][w_idx]/320)
                the_word = wav_embedding_raw[b_idx,start_point:end_point,:]
                aligned_wav_embed = torch.mean(the_word, dim=0)
                wav_aligned[b_idx, w_idx ,:]=aligned_wav_embed
        
        fusion = torch.cat([wav_aligned, gt_embed, asr_word_embed], dim=2)  
        #fusion = fusion.transpose(1, 2) #batch size hidden num_or_word
        fusion = self.fusion_layer(fusion) #shape: batch_size, num_of_word, 768*3 
        #fusion_word = fusion_word.transpose(1, 2) #batch size hidden num_or_word

        uttr_word = torch.mean(fusion, 1)
        wav_embedding = torch.mean(wav_embedding_raw, 1)
        word_embedding_asr = torch.mean(gt_sen_embed,1)
        uttr_embed_asr_wav = torch.cat([word_embedding_asr.unsqueeze(1), wav_embedding.unsqueeze(1)], dim=2)
        uttr_asr = torch.reshape(uttr_embed_asr_wav,(-1,self.text_out_dim+self.ssl_features))
        
        uttr = torch.cat([uttr_asr, uttr_word], dim=1)
        #print(uttr.shape, uttr_word.shape, uttr_asr.shape)
        output_A = self.output_accuracy(uttr)
        output_F = self.output_fluency(uttr)
        output_P = self.output_prosodic(uttr)
        output_T = self.output_total(uttr)
        

        alignment_feature = self.alignment_layer(alignment_feature)
        phonevector = self.phonevector_layer(phonevector)
        fusion_word = torch.cat([wav_aligned, gt_embed, asr_word_embed, alignment_feature, phonevector], dim=2)  
        fusion_word = self.fusion_layer_word(fusion_word)
        fusion_word = fusion_word.transpose(1, 2) #(batch_size, hidden, num_of_word)

        output_w_acc = self.word_acc(fusion_word)
        output_w_stress = self.word_stress(fusion_word)
        output_w_total = self.word_total(fusion_word)
        output_w_acc = output_w_acc.transpose(1,2).squeeze(2) #(batch_size, time, 1)
        output_w_stress = output_w_stress.transpose(1,2).squeeze(2)
        output_w_total = output_w_total.transpose(1,2).squeeze(2)

        return output_A.squeeze(1), output_F.squeeze(1), output_P.squeeze(1), output_T.squeeze(1), output_w_acc, output_w_stress, output_w_total 
    
