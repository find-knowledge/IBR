# IBR
PyTorch code for NAACL 2022 paper:

Interpretable Proof Generation via Iterative Backward Reasoning

曲瀚昊(Hanhao Qu), 曹雨(Yu Cao), 高俊(Jun Gao), 丁亮(Liang Ding), 徐睿峰(Ruifeng Xu)

## Installation:  
IBR is tested on Python 3.6.2.  
pip install -r requirements.txt

## Download the dataset from:  
http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip  
Unzip this folder and create a new folder named data in the home directory and put all data files into it. 

RoBERTa-large:  
Download the pretrained language model from:
https://huggingface.co/roberta-large  
Put it in the home directory.

## Training IBR on DU5:  
Run the main.py.

## Testing IBR on DU5:  
Change do_train setting to False and do_prediction to True in main.py(lines 419 and 421). Run the main.py.

## Testing IBR on Birds-Electricity:  
Change do_train setting to False and do_prediction to True in main.py. Change the data_dir(line 411) setting to './data/birds-electricity'. Run the main.py.

## Training IBR on ParaRules:  
Run the main_natlang.py.

## Testing IBR on ParaRules:  
Change do_train setting to False and do_prediction to True in main_natlang.py(lines 420 and 422). Run the main_natlang.py.

The linked download dataset, which does not include data such as turk-questions-train-mappings.tsv, can be found in the Supplementary Data and Code folder. Also it was recently discovered that in the original model code of ParaRules, the question answer and strategy prediction classification used the RobertaClassificationHead structure (it is a two linear layer structure, the main experiment in the paper is a one linear layer), so to reproduce the results of the experiments on ParaRules, the model files are also in the Supplementary Data and Code folder ( Only for reproducing the ParaRules results using the uploaded model, the model training is still using the previously uploaded code).

## Trained Models
We release our trained IBR models on depth-5 dataset and ParaRules dataset [here](https://drive.google.com/file/d/1hv5Yk1cRKXL0oF2HTnKxrhWECtyydZkX/view?usp=sharing).
