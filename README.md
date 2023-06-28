# pronunciation_assessment

requirement

conda create -n chatbot python=3.9  
conda activate chatbot  
pip install fairseq   
pip install soundfile  
pip install -U openai-whisper  
pip install transformers  
pip install num2words   
pip install pyctcdecode  
pip install https://github.com/kpu/kenlm/archive/master.zip  
pip install spacy==2.3.0 (require 2.x version)
pip install levenshtein  
pip install nltk  

In api.py, change the filepath to the path of wavfle. 

results = prediced_one_file(filepath,  whisper_model_s, whisper_model_w, roberta, model, aligner, device)


pretrained_model: https://drive.google.com/file/d/1m29M4dlvwWIwbvWDcV8yU_wkvi79bhAh/view?usp=sharing


Reproduce testing result:
Download speechocean762 dataset: https://www.openslr.org/101

```
test_alignedmodel3_gt.py --datadir /path/to/speechocean/wav --datalist /path/to/speechocean762_test.txt --ckptdir /path/to/chkp/dir
```
Put the "scores.json" to ./data directory, or change the path in line 32 "f = open('./data/scores.json')" to where you put the scores.json

the Results will be saved in ./Results/{ckptdir_name}_speechocean762_test_gtb.txt  
(See speechocean762_test_gtb.txt  for example result.)


Calculate evalutation score:
```
evaluation_word_alignedmodel.py
```
change the path on line 24, 27, and 157
