import os
import gc
import argparse
import torch
import torch.nn as nn
import fairseq
import string
import numpy as np
import torchaudio
import soundfile as sf
import whisper
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor
from utils import *
from model import PronunciationPredictor
import time

gc.collect()
torch.cuda.empty_cache()

def prediced_one_file(filepath, whisper_model_s, whisper_model_w, roberta, model, aligner, device):

    results = {}
    results['wavname'] = filepath.split('/')[-1]
    with torch.no_grad():

        wav, sr = torchaudio.load(filepath)
        #resample audio recordin to 16000Hz
        if sr!=16000:
            transform = torchaudio.transforms.Resample(sr, 16000)
            wav = transform(wav)
            sr = 16000
            
        if wav.shape[0]!=1:
            wav = torch.mean(wav,0)                                                                                          
            wav = torch.reshape(wav, (1, -1))

        wav = wav.to(device)

        try:
            sen_asr_s_ori = get_transcript(filepath, whisper_model_s)
            sen_asr_s = remove_pun_except_apostrophe(sen_asr_s_ori).lower()
            sen_asr_s = remove_pun_except_apostrophe(sen_asr_s).lower()
            sen_asr_s = convert_sentence(sen_asr_s)
            sen_asr_w_ori = get_transcript(filepath, whisper_model_w)
            sen_asr_w = remove_pun_except_apostrophe(sen_asr_w_ori).lower()
            sen_asr_w = convert_sentence(sen_asr_w)
            align_s = get_alignment(filepath, sen_asr_s+'.', aligner)
            align_w = get_alignment(filepath, sen_asr_w+'.', aligner)
            gt_word_list, asr_word_list, alignment_features, phone_vector = align_results(align_s, align_w)   

            start = [[word[1] for word in align_s]]
            end = [[word[2] for word in align_s]]
            num_of_token = len(gt_word_list)
            gt_sen = ' '.join([sublist[0] for sublist in gt_word_list])

            gt_vec =  get_roberta_word_embed(gt_word_list, num_of_token, roberta)
            asr_word_vec = get_roberta_word_embed(asr_word_list, num_of_token, roberta)
            gt_sen_vec = get_embed(gt_sen, roberta)
            asr_word_vec = torch.from_numpy(asr_word_vec).to(device).float()
            gt_sen_vec = gt_sen_vec.to(device).float()
            gt_vec = torch.from_numpy(gt_vec).to(device).float()
            alignment_features = torch.from_numpy(alignment_features).to(device).float()
            phone_vector = torch.from_numpy(phone_vector).to(device).float()
        
            score_A, score_F, score_P, score_T, w_acc, w_stress, w_total = model(wav, asr_word_vec, gt_sen_vec, gt_vec, alignment_features, phone_vector, start, end)
            
            score_A = score_A.cpu().detach().numpy()[0]
            score_F = score_F.cpu().detach().numpy()[0]
            score_P = score_P.cpu().detach().numpy()[0]
            score_T = score_T.cpu().detach().numpy()[0] 
            w_a = w_acc.cpu().detach().numpy()[0]
            w_s = w_stress.cpu().detach().numpy()[0]
            w_t = w_total.cpu().detach().numpy()[0]
  

            results['uttr_acc'] = score_A
            results['uttr_fluency'] = score_F
            results['uttr_prosodic'] = score_P
            results['uttr_total'] = score_T

            results['word_acc'] = w_a
            results['word_stress'] = w_s
            results['word_total'] = w_t
            results['start'] = start[0]
            results['end'] = end[0]
            results['tokens'] = gt_word_list
            results['transcript_S'] = sen_asr_s_ori
            results['transcript_W'] = sen_asr_w_ori

            torch.cuda.empty_cache()

            return results
        except Exception as e:
            print('ERROR:', e)
            return 
    
def main():
    
    ssl_path = './fairseq/hubert_base_ls960.pt'
    my_checkpoint_dir = './model_alignedword3_v4_b_r3'

    roberta = RobertaModel.from_pretrained('./fairseq', checkpoint_file='model.pt')
    roberta.eval()
    whisper_model_w = whisper.load_model("base.en")
    whisper_model_s = whisper.load_model("medium.en")

    wav2vec_model = 'arijitx/wav2vec2-xls-r-300m-bengali'
    input_wavs_sr = 16000
    aligner = Wav2Vec2Aligner(wav2vec_model, input_wavs_sr, True)

    SSL_OUT_DIM = 768
    TEXT_OUT_DIM = 768

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    ssl_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
    ssl_model = ssl_model[0]
   
    model = PronunciationPredictor(ssl_model, SSL_OUT_DIM, TEXT_OUT_DIM).to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'PRO'+os.sep+'best')))

    file_dir = '/home/yuwchen/pronunciation/data/wav'
    filepath_list = get_filepaths(file_dir)

    for filepath in filepath_list:
        s = time.time()
        
        results = prediced_one_file(filepath,  whisper_model_s, whisper_model_w, roberta, model, aligner, device)
        print('Process time:', time.time()-s)
        print(results)
 

if __name__ == '__main__':
    main()
