import os
import gc
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
import whisper
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor
import string
import num2words
from Levenshtein import ratio
import nltk
from nltk.corpus import cmudict
from difflib import SequenceMatcher

def fit_one_hot(inputlist):
    mapping = {}
    for i in range(len(inputlist)):
        mapping[inputlist[i]]=i
    return mapping

nltk.download('cmudict')

all_phonemes = set()
entries = cmudict.entries()
for entry in entries:
    phonemes = entry[1]
    all_phonemes.update(phonemes)
all_phonemes = list(all_phonemes)
all_phonemes.append('')
all_phonemes = fit_one_hot(all_phonemes)
cmudict_dict = {entry[0].lower(): entry[1] for entry in entries}

def reshape_format(input_embed):
    length =  len(input_embed)
    input_embed = np.asarray(input_embed)
    input_embed = np.reshape(input_embed,(1, length, 1))
    return input_embed

def get_phone_vector(phone_list):
    num_of_word = len(phone_list)
    num_of_phone = len(all_phonemes.keys())
    phone_vector = np.zeros((1, num_of_word, num_of_phone))
    for word_idx in range(num_of_word):
        the_phone_list = phone_list[word_idx]
        the_phone_vector = np.zeros((num_of_phone, ))
        for phone in the_phone_list:
            the_phone_vector[all_phonemes[phone]]+=1
        phone_vector[0, word_idx,:] = the_phone_vector

    return phone_vector

def get_standard_pronunciation(word):
    entries = cmudict.entries()
    for entry in entries:
        if entry[0].lower() == word.lower():
            return entry[1]
    return ''


def get_phone_list(word_list):

    phone_list = []
    for word_position in word_list:
        the_phone_list = []
        for word in word_position:
            phone = cmudict_dict.get(word.lower(), '')
            the_phone_list.extend(phone)
        phone_list.append(the_phone_list)
    return phone_list


def get_phone_features(gt_word_list, asr_word_list):

    gt_length = len(gt_word_list)
    gt_phone_list = get_phone_list(gt_word_list)
    asr_phone_list = get_phone_list(asr_word_list)

    gt_phone_vector = get_phone_vector(gt_phone_list)
    asr_phone_vector = get_phone_vector(asr_phone_list)
    
    phone_distance = []
    phone_count = []
    for word_idx in range(gt_length):
        the_distance = SequenceMatcher(None, gt_phone_list[word_idx],asr_phone_list[word_idx])
        phone_distance.append(the_distance.ratio())
        if len(asr_phone_list[word_idx])!=0:
            phone_count.append(len(gt_phone_list[word_idx])/len(asr_phone_list[word_idx]))
        else:
            phone_count.append(0)

    phone_vector = np.concatenate((gt_phone_vector, asr_phone_vector), axis=2)

    return phone_distance, phone_vector, phone_count

def align_results(result_gt, result_asr):

    gt_length = len(result_gt)
    asr_length =  len(result_asr)

    asr_distance = []
    gt_word_list = []
    asr_word_list = []
    duration_gt = []
    duration_asr = []
    time_diff_start = [] 
    time_diff_end = []
    gt_word_prob = []
    asr_word_prob = []
    wav_length =  result_gt[-1][2]

    for gt_idx in range(gt_length):

        gt_word = result_gt[gt_idx][0]
        gt_start = result_gt[gt_idx][1]
        gt_end = result_gt[gt_idx][2]
        gt_prob = result_gt[gt_idx][3]

        #duration_gt.append(gt_end-gt_start)
        duration_gt.append((gt_end-gt_start)/wav_length)
        gt_word_prob.append(gt_prob)

        asr_word_all = []
        asr_all_prob = 0
        asr_start_flag = True
        the_asr_start = 0
        the_asr_end = result_asr[-1][2]   
        for asr_idx in range(asr_length):
            asr_word = result_asr[asr_idx][0]
            asr_start = result_asr[asr_idx][1]
            asr_end = result_asr[asr_idx][2]   
            asr_prob = result_asr[asr_idx][3] 
            if gt_end <= asr_start:
                break
            if gt_start >= asr_end:
                continue
            if max(gt_start, asr_start) <= min(gt_end, asr_end):
                asr_word_all.append(asr_word)
                asr_all_prob += asr_prob
                the_asr_end = asr_end
                if asr_start_flag:
                    the_asr_start = asr_start
                    asr_start_flag= False

        #duration_asr.append(the_asr_end - the_asr_start)
        #time_diff_start.append(the_asr_start - gt_start)
        #time_diff_end.append(the_asr_end - gt_end)
        duration_asr.append((the_asr_end - the_asr_start)/wav_length)
        time_diff_start.append((the_asr_start - gt_start)/wav_length)
        time_diff_end.append((the_asr_end - gt_end)/wav_length)

        asr_word_prob.append(asr_all_prob)

        gt_word_list.append([gt_word])
        asr_word_list.append(asr_word_all)
        the_distance = ratio(gt_word, ' '.join(asr_word_all))*10
        asr_distance.append(the_distance)
        
    align_word_count = []
    for word_idx in range(gt_length):
        count = len(asr_word_list[word_idx])
        align_word_count.append(count)

    phone_distance, phone_vector, phone_count = get_phone_features(gt_word_list, asr_word_list)

    asr_distance = reshape_format(asr_distance)
    align_word_count = reshape_format(align_word_count)
    duration_gt = reshape_format(duration_gt)
    duration_asr = reshape_format(duration_asr)
    time_diff_start = reshape_format(time_diff_start)
    time_diff_end = reshape_format(time_diff_end)
    gt_word_prob = reshape_format(gt_word_prob)
    asr_word_prob = reshape_format(asr_word_prob)
    phone_distance = reshape_format(phone_distance)
    phone_count = reshape_format(phone_count)

    alignment_features = np.concatenate((asr_distance, align_word_count, duration_gt, duration_asr, time_diff_start, time_diff_end, gt_word_prob, asr_word_prob, phone_distance, phone_count), axis=2)

    return gt_word_list, asr_word_list, alignment_features, phone_vector



def get_roberta_word_embed(asr_word_list, num_of_token, roberta):

    sen_vec = np.zeros((1, num_of_token, 768))
    for w_idx in range(num_of_token):
        the_sen = ' '.join(asr_word_list[w_idx])
        if the_sen=='':
            continue
        doc = roberta.extract_features_aligned_to_words(the_sen)
        the_sen_vec = np.zeros((1, 768))
        for tok in doc:
            if str(tok)=='<s>' or str(tok)=='</s>':
                continue
            the_vec = tok.vector.detach().numpy()
            the_sen_vec[0,:] += the_vec
        the_sen_vec /= len(asr_word_list[w_idx])
        sen_vec[0,w_idx,:] = the_sen_vec

    return sen_vec 

def get_alignment(filepath, sen, aligner, return_emission=False):
    try:
        if not return_emission:
            align_result = aligner.align_data([filepath], [sen], '', return_emission)[0]
            return align_result
        else:
            alignment_result = aligner.align_data([filepath], [sen], '', return_emission)[0]
            align_result = alignment_result[0]
            emission = alignment_result[1]
            return align_result, emission            
    except:
        return None

def convert_num_to_words(utterance):
      utterance = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in utterance.split()])
      return utterance

def convert_sentence(sen):
    try:
        int(sen.replace(' ',''))
        sen = ' '.join([char for char in sen])
        sen = convert_num_to_words(sen)
    except:
        sen = convert_num_to_words(sen)
    return sen

def remove_pun_except_apostrophe(input_string):
    # create a translation table with all punctuation characters except apostrophe
    translator = str.maketrans('', '', string.punctuation.replace("'", ""))
    # remove all punctuation except apostrophe using the translation table
    output_string = input_string.translate(translator).replace('  ',' ')
    return output_string

def remove_pun(input_string):
    input_string = "".join([char for char in input_string if char not in string.punctuation])
    return input_string

def get_embed(sen, roberta):
    tokens = roberta.encode(sen)
    last_layer_features = roberta.extract_features(tokens).detach()
    return last_layer_features

def get_transcript(filepath, whisper_model):
    result = whisper_model.transcribe(filepath, fp16=False)
    transcript = result['text']
    return transcript

def get_filepaths(directory):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 

def get_embed_align(sen, roberta):
    
    doc = roberta.extract_features_aligned_to_words(sen)
    sen_vec = []
    tokens = []
    pre = ''
    for tok in doc:
        if str(tok)=='<s>' or str(tok)=='</s>':
            continue
        if ('\'' in str(tok)) or ( pre =='can' and str(tok)=='not'):
            previous_vec = sen_vec[-1]
            the_vec = tok.vector.detach().numpy()
            sen_vec[-1] = previous_vec + the_vec
            tokens[-1] = pre+str(tok)
            pre = str(tok)
        elif str(tok) == ' ':
            pass
        else:
            the_vec = tok.vector.detach().numpy()
            sen_vec.append(the_vec)
            pre = str(tok) 
            tokens.append(str(tok))

    sen_vec = np.asarray(sen_vec)
    sen_vec = np.reshape(sen_vec, (1,-1, sen_vec.shape[1]))
    sen_vec = torch.from_numpy(sen_vec)

    return sen_vec, tokens

class Wav2Vec2Aligner:
    def __init__(self, model_name, input_wavs_sr, cuda):
        self.cuda = cuda
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()
        if self.cuda:
            self.model.to(device="cuda")
        self.processor = AutoProcessor.from_pretrained(model_name)
        blank_id = 0
        vocab = list(self.processor.tokenizer.get_vocab().keys())
        for i in range(len(vocab)):
            if vocab[i] == "[PAD]" or vocab[i] == "<pad>":
                blank_id = i
        print("Blank Token id [PAD]/<pad>", blank_id)
        self.blank_id = blank_id

    def speech_file_to_array_fn(self, wav_path):
        speech_array, sampling_rate = torchaudio.load(wav_path)
        if speech_array.shape[0]>1:
            speech_array = torch.mean(speech_array, 0)
            speech_array = torch.reshape(speech_array,(1,-1))

        if sampling_rate!=16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech = resampler(speech_array).squeeze().numpy()
        else:
            speech = speech_array.squeeze().numpy()

        return speech

    def get_emission(self, wavpath):

        speech_array = self.speech_file_to_array_fn(wavpath)
        inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        if self.cuda:
            inputs = inputs.to(device="cuda")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        #emissions = torch.log_softmax(logits, dim=-1)
        emissions = logits
        emission = emissions[0].cpu().detach()
        return emission


    def align_single_sample(self, item, return_emission=False):
        blank_id = self.blank_id
        transcript = "|".join(item["sent"].split(" "))
        if not os.path.isfile(item["wav_path"]):
            print(item["wav_path"], "not found in wavs directory")

        speech_array = self.speech_file_to_array_fn(item["wav_path"])
        inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        if self.cuda:
            inputs = inputs.to(device="cuda")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        emissions_ori = torch.log_softmax(logits, dim=-1)
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()

        # get labels from vocab
        labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
            :-1
        ]  # logits don't align with the tokenizer's vocab

        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])

        def get_trellis(emission, tokens, blank_id=0):
            """
            Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
            that represents the probabilities of each source token being at a certain time step
            """
            num_frames = emission.size(0)
            num_tokens = len(tokens)

            # Trellis has extra diemsions for both time axis and tokens.
            # The extra dim for tokens represents <SoS> (start-of-sentence)
            # The extra dim for time axis is for simplification of the code.
            trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
            trellis[:, 0] = 0
            for t in range(num_frames):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            return trellis

        trellis = get_trellis(emission, tokens, blank_id)

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, tokens, blank_id=0):
            """
            Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path
            """
            # Note:
            # j and t are indices for trellis, which has extra dimensions
            # for time and tokens at the beginning.
            # When referring to time frame index `T` in trellis,
            # the corresponding index in emission is `T-1`.
            # Similarly, when referring to token index `J` in trellis,
            # the corresponding index in transcript is `J-1`.
            j = trellis.size(1) - 1
            t_start = torch.argmax(trellis[:, j]).item()

            path = []
            for t in range(t_start, 0, -1):
                # 1. Figure out if the current position was stay or change
                # Note (again):
                # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
                # Score for token staying the same from time frame J-1 to T.
                stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
                # Score for token changing from C-1 at T-1 to J at T.
                changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

                # 2. Store the path with frame-wise probability.
                prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
                # Return token index and time index in non-trellis coordinate.
                path.append(Point(j - 1, t - 1, prob))

                # 3. Update the token
                if changed > stayed:
                    j -= 1
                    if j == 0:
                        break
            else:
                raise ValueError("Failed to align")
            return path[::-1]

        path = backtrack(trellis, emission, tokens, blank_id)

        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return f"{self.label}\t{self.score:4.5f}\t{self.start*20:5d}\t{self.end*20:5d}"

            @property
            def length(self):
                return self.end - self.start

        def merge_repeats(path):
            """
            Merge repeated tokens into a single segment. Note: this shouldn't affect repeated characters from the
            original sentences (e.g. `ll` in `hello`)
            """
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        transcript[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score,
                    )
                )
                i1 = i2
            return segments

        segments = merge_repeats(path)
        the_result = []
        word = ''
        first_char=True
        score = 0
        for seg in segments:
            parts = str(seg).split('\t')
            character = parts[0]
            the_score = float(parts[1])
            start_ms = parts[2]
            end_ms = parts[3]
            start_point = int(start_ms)*0.001*16000
            end_point = int(end_ms)*0.001*16000
            w_e = end_point
            if character !='|':
                word = word+character
                score+=the_score
                if first_char:
                    w_s = start_point 
                    first_char=False
            else: # reset
                the_result.append((word, int(w_s), int(w_e), score))
                word = ''
                w_s = 0
                w_e = 0
                score = 0
                first_char = True
        the_result.append((word, int(w_s), int(w_e), score))

        if not return_emission:
            return the_result
        else:
            emissions_ori = torch.reshape(emissions_ori, (1, -1, 112))
            return [the_result, emissions_ori]

    def align_data(self, wav_dir, text_file, output_dir, return_emission):

        assert len(wav_dir)==len(text_file)

        items = []
        for wav_idx in range(len(wav_dir)):
            wav_path = wav_dir[wav_idx]
            sentence = text_file[wav_idx]
            wav_name = os.path.basename(wav_path).replace('.wav','.txt')
            out_path = os.path.join(output_dir, wav_name)
            items.append({"sent": sentence, "wav_path": wav_path, "out_path": out_path})

        result = []
        for item in items:
            the_result = self.align_single_sample(item, return_emission)
            result.append(the_result)
        return result     
