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

print("READING THE DATA")

'''
with open("data/all_session_tracks_train.pkl", 'rb') as f:
	train_tracks = pickle.load(f)
	train_tracks = train_tracks[0:N]

with open("data/all_session_skips_train.pkl", 'rb') as f:
	train_skips = pickle.load(f)
	train_skips = train_skips[0:N]
'''

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

#train_dataset = SpotifyDataset(train_tracks, train_skips, track_vocab, bert=False, bert_mask_proportion = 0.2, skip_pred=SKIP, padding=False)
valid_dataset = SpotifyDataset(test_tracks, test_skips, track_vocab, bert=False, bert_mask_proportion = 0.2, skip_pred=SKIP, padding=False)

#train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn =custom_collate_fn)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn =custom_collate_fn)

#BERT AUG SEQ MODEL PREDICTIONS
bert_model = torch.load('data/model_base_bert_0.2_1e-4_256_dim_v2.pth')
bert_model.fc = nn.Identity()

learning_rate = 1e-5
model = torch.load('output/model_bert_augmented_seq_embed_1e-5_256_dim.pth')
model.to(device)
model.eval()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

outputs = []
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

preds = []
labels = []

print("EVALUATING SEQ")
with torch.no_grad():
	for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(valid_loader):

		input_sequence = input_sequence.to(device)

		bert_output = bert_model(input_sequence)
		bert_output = rearrange(bert_output, 'n s t -> s n t')

		label = label_sequence.cuda()
		output_tokens, outputs = model.greedy_decoder(model, input_sequence, bert_output, 10, input_sequence[:,-1])

		preds.append(output_tokens)
		labels.append(label)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print(preds.shape)
print(labels.shape)

print("SAVING SEQ")
np.save('transformer_bert_aug_seq_preds.npy', preds)
np.save('transformer_bert_aug_seq_labels.npy', labels)


'''
#SEQ MODEL PREDICTIONS
learning_rate = 1e-5
model = torch.load('output/model_bert_transformer_seq_embed_1e-5_256_dim.pth')
model.to(device)
model.eval()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

outputs = []
if SKIP:
	criterion = nn.CrossEntropyLoss()
else:
	criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

preds = []
labels = []

print("EVALUATING SEQ")
with torch.no_grad():
	for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(valid_loader):
		input_sequence = input_sequence.to(device)
		label = label_sequence.cuda()
		output_tokens, outputs = model.greedy_decoder(model, input_sequence, 10, input_sequence[:,-1])
		preds.append(output_tokens)
		labels.append(label)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print(preds.shape)
print(labels.shape)

print("SAVING SEQ")
np.save('transformer_seq_preds.npy', preds)
np.save('transformer_seq_labels.npy', labels)

#SKIP MODEL PREDICTIONS
learning_rate = 1e-6
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

		preds.append(outputs)
		labels.append(label)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)

preds = preds.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

print(preds.shape)
print(labels.shape)

print("SAVING SKIP")
np.save('transformer_skip_preds.npy', preds)
np.save('transformer_skip_labels.npy', labels)
'''

