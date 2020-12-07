import pandas as pd
import numpy as np
import math
import pickle
import torch
import random
import torch.nn as nn
import torch.optim as optim
import time
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from models.BertSeqTransformer import StandardTransformer
from datasets.SpotifyDataset import SpotifyDataset, bert_collate_fn, custom_collate_fn

N=100000
torch.manual_seed(1)
EPOCHS = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

#AVERAE METER FROM BD4H UTIL FUNCTIONS
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
	
	"""Computes the average accuracy of the predicted skip sequence"""
	
	seq_len = target.shape[1]
	correct = output.eq(target)
	correct = correct.sum(axis=1) * 1.0
	acc = correct / seq_len
	return acc

def mean_average_accuracy(output, target):
	
	"""Computes the  mean average accuracy of the predicted skip sequence up till the 10th seq position"""

	T = output.shape[1]
	batch_size = target.shape[0]
	output = torch.argmax(output, dim=2)

	A_i = torch.zeros(batch_size, T)
	L_i = torch.zeros(batch_size, T)

	for i in range(1,T+1):
		
		A_i[:,i-1] = accuracy(output[:,0:i], target[:,0:i])
		pred_i = output[:,i-1]
		target_i = target[:,i-1]
		L_i[:,i-1] = pred_i.eq(target_i)*1.0
		A_i[:,i-1] = A_i[:,i-1]*L_i[:,i-1]

	AA = A_i.sum(dim =1) / T

	return torch.sum(AA) / batch_size

def accuracy_at_k(output, target):
    
    output = output.to(device)
    target = target.to(device)
    
    T = output.shape[1]
    batch_size = target.shape[0]
    output = torch.argmax(output, dim=2)
    acc = torch.zeros(T)

    for i in range(T):        
        acc[i] = torch.mean(accuracy(output[:,i].reshape(batch_size,1), target[:,i].reshape(batch_size,1)))
        
    return acc

print("READING THE DATA")

with open("data/all_session_tracks_test.pkl", 'rb') as f:
	test_tracks = pickle.load(f)
	test_tracks = test_tracks[0:N]

with open("data/all_session_skips_test.pkl", 'rb') as f:
	test_skips = pickle.load(f)
	test_skips = test_skips[0:N]

with open("data/track_vocabs.pkl", 'rb') as f:
	track_vocab = pickle.load(f)

INPUT_SIZE = len(track_vocab)
OUTPUT_SIZE = len(track_vocab)
PAD_IDX = track_vocab['pad']
MAX_LEN = 20
SKIP = False
PAD_MASK = 1
BATCH_SIZE = 256

valid_dataset = SpotifyDataset(test_tracks, test_skips, track_vocab, bert=False, bert_mask_proportion = 0.2, skip_pred=SKIP, padding=False)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn =custom_collate_fn)

#BERT AUG SEQ MODEL PREDICTIONS
bert_model = torch.load('data/model_base_bert_0.2_1e-4_256_dim_v2.pth')
bert_model.fc = nn.Identity()

learning_rate = 1e-5
model = torch.load('output/model_bert_augmented_seq_embed_1e-5_256_dim.pth')
model.to(device)
model.eval()

preds = None
labels = None
preds = []
labels = []

print("EVALUATING BERT SEQ")
with torch.no_grad():
	for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(valid_loader):

		input_sequence = input_sequence.to(device)

		bert_output = bert_model(input_sequence)
		bert_output = rearrange(bert_output, 'n s t -> s n t')

		label = label_sequence.cuda()
		output_tokens, outputs = model.greedy_decoder(model, input_sequence, bert_output, 10, input_sequence[:,-1])
		out_preds = torch.argmax(outputs, dim=2)
		preds.append(out_preds)
		labels.append(label)

		if (batch_idx % 100) == 0:
			acc = mean_average_accuracy(outputs, label)
			acc_k = accuracy_at_k(outputs, label)
			print(acc_k)
			print(acc)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print("SAVING BERT SEQ")
np.save('transformer_bert_aug_seq_preds_v2.npy', preds)
np.save('transformer_bert_aug_seq_labels_v2.npy', labels)


#SEQ MODEL PREDICTIONS
learning_rate = 1e-5
model = None
model = torch.load('output/model_bert_transformer_seq_embed_1e-5_256_dim.pth')
model.to(device)
model.eval()

preds = None
labels = None
preds = []
labels = []

print("EVALUATING SEQ")
with torch.no_grad():
	for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(valid_loader):
		input_sequence = input_sequence.to(device)
		label = label_sequence.cuda()
		output_tokens, outputs = model.greedy_decoder(model, input_sequence, 10, input_sequence[:,-1])

		out_preds = torch.argmax(outputs, dim=2)
		preds.append(out_preds)
		labels.append(label)
		
		if (batch_idx % 100) == 0:
			acc = mean_average_accuracy(outputs, label)
			acc_k = accuracy_at_k(outputs, label)
			print(acc_k)
			print(acc)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print("SAVING SEQ")
np.save('transformer_seq_preds_v2.npy', preds)
np.save('transformer_seq_labels_v2.npy', labels)

#SKIP MODEL PREDICTIONS
learning_rate = 1e-6
model = None
model = torch.load('output/model_bert_transformer_skip_embed_1e-6.pth')
model.to(device)
model.eval()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

preds = []
labels = []

print("EVALUATING SKIP")
with torch.no_grad():
	for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(valid_loader):
		input_sequence = input_sequence.to(device)
		label_sequence = label_sequence.cuda()
		label = label_skips.cuda()
		input_skips = input_skips.cuda()
		outputs = model(input_sequence, label_sequence, input_skips)
		#out_pred = torch.argmax(outputs, dim=2)
		preds.append(outputs)
		labels.append(label)
		
		if (batch_idx % 100) == 0:
			acc = mean_average_accuracy(outputs, label)
			acc_k = accuracy_at_k(outputs, label)
			print(acc_k)
			print(acc)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print("SAVING SKIP")
np.save('transformer_skip_preds_v2.npy', preds)
np.save('transformer_skip_labels_v2.npy', labels)
