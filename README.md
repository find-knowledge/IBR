# IBR
Download the dataset from:  
http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip  
Unzip this folder and create a new folder named data in the home directory and put all data files into it. 

RoBERTa-large:  
Download the pretrained language model from:
https://huggingface.co/roberta-large  
Put it in the home directory.

Training IBR on DU5:  
Run the main.py.

Testing IBR on DU5:  
Change do_train setting to False and do_prediction to True in main.py(lines 419 and 421). Run the main.py.

Testing IBR on Birds-Electricity:  
Change do_train setting to False and do_prediction to True in main.py. Change the data_dir(line 411) setting to './data/birds-electricity'. Run the main.py.

Training IBR on ParaRules:  
Run the main_natlang.py.

Testing IBR on ParaRules:  
Change do_train setting to False and do_prediction to True in main_natlang.py(lines 420 and 422). Run the main_natlang.py.
