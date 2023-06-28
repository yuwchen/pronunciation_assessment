import os
import gc
import math
import json
import argparse
import torch
import torch.nn as nn
import fairseq
import string
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
import whisper
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor
from utils import *
from model import PronunciationPredictor

gc.collect()
torch.cuda.empty_cache()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, default='./fairseq/hubert_base_ls960.pt', help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', default='./data/wav', type=str, help='Path of your DATA/ directory')
    parser.add_argument('--datalist', default='./data/speechocean762_test.txt', type=str, help='')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')

    f = open('./data/scores.json') #path to speechocean scores.json
    gt_data = json.load(f)

    args = parser.parse_args()
    
    ssl_path = args.fairseq_base_model
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    datalist = args.datalist

    roberta = RobertaModel.from_pretrained('./fairseq', checkpoint_file='model.pt')
    roberta.eval()
    whisper_model_s = whisper.load_model("base.en")

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

    print('Loading data')

    validset = open(datalist,'r').read().splitlines()
    outfile = my_checkpoint_dir.split("/")[-1]+'_'+datalist.split('/')[-1].replace('.txt','_alignedword_gtb_nt.txt')

    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    prediction = open(os.path.join(output_dir, outfile), 'w')
    
    print('Starting prediction')
    for filename in tqdm(validset):
        
        with torch.no_grad():
            if datalist is not None:
                filepath = os.path.join(datadir, filename)
            else:
                filepath=filename
            wav, sr = torchaudio.load(filepath)
            #resample audio recordin to 16000Hz
            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
                sr = 16000
            
            wav = wav.to(device)

            sen_asr_s = [word.lower() for word in gt_data[filename.replace('.wav','')]['words']]
            sen_asr_s = ' '.join(sen_asr_s)
            
            sen_asr_s = remove_pun_except_apostrophe(sen_asr_s)
            sen_asr_s = convert_sentence(sen_asr_s)

            sen_asr_w = remove_pun_except_apostrophe(get_transcript(filepath, whisper_model_s)).lower()
            sen_asr_w = convert_sentence(sen_asr_w)

            try:
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
                
                start  = [str(x/sr) for x in start[0]]
                end = [str(x/sr) for x in end[0]]
                start = ','.join(start)
                end = ','.join(end)
                
                w_a = ','.join([str(num) for num in w_a])
                w_s = ','.join([str(num) for num in w_s])
                w_t = ','.join([str(num) for num in w_t])

                valid = 'T'
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; Start:{}; End:{}".format(filename, score_A, score_F, score_P, score_T, valid, gt_word_list, asr_word_list, w_a, w_s, w_t, start, end)
                print(output)
                prediction.write(output+'\n')

            except Exception as e:
                print(e)
                valid = 'F'
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; Start:{}; End:{}".format(filename, '', '', '', '', valid, '', '', '', '', '', '', '')
                prediction.write(output+'\n')
                continue
               

            torch.cuda.empty_cache()

            
 

if __name__ == '__main__':
    main()
