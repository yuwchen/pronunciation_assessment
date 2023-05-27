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

In api.py, change the input of the filepath. 

results = prediced_one_file(filepath,  whisper_model_s, whisper_model_w, roberta, model, aligner, device)


pretrained_model: https://drive.google.com/file/d/1m29M4dlvwWIwbvWDcV8yU_wkvi79bhAh/view?usp=sharing
