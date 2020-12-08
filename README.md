# Sessions Based Music Recommendations Using Deep Learning

This is the repo for Final project for CS7643-Deep Learning

Authors
Ahmad Khan: akhan361@gatech.edu
Tianyuan Cui: tcui33@gatech.edu
Sagar Arora: sarora83@gatech.edu

## Getting the Source Data

The data used for this work is from CrowdAI Spotify skip challenge
https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge

We trimmed the data to only include 103k most frequent tracks(items) only to expedite model iterations
You may download data used for this project in this google drive https://drive.google.com/file/d/1UrZKi6goybLZZlRA9pUelOFY-VTsqJFE/view


### How to run
#### Deep reinforcement learning (Tianyuan's Code Contribution)
Download the data to ./DRL/data 
The skip prediction code can be found in .DRL/model/RL_based-binary.ipynb
The sequence prediction code can be found in .DRL/model/RL_Based_seq.ipynb

#### LSTM based architecture (Sagar's Code Contribution)
Download the data to ./data
The LSTM based architctures can be run using the notebook spotify_lstm_architectures.ipynb

#### Transformer Based Architectures (Ahmad's Code Contribution)
Download the data to ./data
create  environment: `conda env create -f environment.yml`
and then `source activate bd4h_project`

The transformer models are all in `model` folder.
1) `SeqTransformer.py` this is the Standard Transformer Track Sequence Prediction Model
2) `SkipTransformer.py`this is the Standard Transformer Track Skip Prediction Model
3) `BertAugmentedTransformer.py` this is the Bert Augmented Transformer mentioned in our paper (uses `CustomizedTransformer.py`)
4) `BertAugmentedTransformer.py` this is the Bert Augmented Transformer with DropNet mentioned in our paper (uses `CustomizedTransformerDropnet.py`)
5) `BertTransformer.py` this is the Bert Encoder model used for both pretrainining and fine tuning a modified BERT For the recommendation task
6)  `CustomizedTransformer.py` this is the custom Torch Transformer Layers built to be used in the Bert Augmented Transformer (modified from original Torch source code)
7)  `CustomizedTransformerDropNet.py` this is the custom Torch Transformer Layers with DropNet added built to be used in the Bert Augmented Transformer (modified from original torch source code)

The training scripts for the models are all in main folder.
1) `train_transformer_seq.py` trains the Standard Transformer track sequence prediction model
2) `train_transformer_skip.py` trains the Standard Transformer track skip prediction model
3) `train_seq_bert_augmented.py` trains the Bert Augmented Transformer track sequence prediction model
4) `train_skip_bert_augmented.py` trains the Bert Augmented Transformer track skip prediction model
5) `train_seq_bert_augmented_dropnet.py` trains the Bert Augmented Transformer with DropNet for the track sequence prediction model
6) `train_bert_pretrain.py` preetrains the BERT model with 20% masking for the Track corpus
7) `train_bert_finetune_skip.py` trains the BERT finetuned model afteer pretraining for the track skip problem
8) `train_bert_finetune_seq.py` trains the BERT finetuned skip model after pretraining for the track skip problem

Additionally there is a little bit of modeling analysis done in the following notebook:
1) `spotify_model_analysis_skip_seq.ipynb`

Finally the following script was used to create the sampled dataset of 3.3M sessions with 103K unique track ids used in all models
1) `utils/datapreprocessing.py`


####
