import math
import time
import numpy as np
import pandas as pd
import glob
import pickle
import random
import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

# Tqdm progress bar
from tqdm import tqdm_notebook

from models.Transformer import Transformer
from datasets.SpotifyDataset import SpotifyDataset

##FOR BERT LM MASKED PRETRAINING AND EVALUATION
def train(model, dataloader, optimizer, criterion, scheduler = None):

    model.train()
    total_loss = 0.0

    progress_bar = tqdm_notebook(dataloader, ascii = True)

    for batch_idx, (inputs, labels) in enumerate(dataloader):

        outputs = model(inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        labels = labels.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss

        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))

    return total_loss, total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar 
        progress_bar = tqdm_notebook(dataloader, ascii = True)
        for batch_idx, (inputs, labels) in enumerate(dataloader):

            outputs = model(inputs)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)
            total_loss += loss

            progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

#Read the data
with open("data/all_session_tracks_train.pkl", 'rb') as f:
    train_tracks = pickle.load(f)

with open("data/all_session_skips_train.pkl", 'rb') as f:
    train_skips = pickle.load(f)

with open("data/all_session_tracks_test.pkl", 'rb') as f:
    test_tracks = pickle.load(f)

with open("data/all_session_skips_test.pkl", 'rb') as f:
    test_skips = pickle.load(f)

with open("data/track_vocabs.pkl", 'rb') as f:
    track_vocab = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

INPUT_SIZE = len(track_vocab)
OUTPUT_SIZE = len(track_vocab)
PAD_IDX = track_vocab['pad']
BATCH_SIZE = 256
MAX_LEN = 20
EPOCHS = 10

train_dataset = SpotifyDataset(train_tracks, train_skips, track_vocab, bert=True, bert_mask_proportion = 0.2)
valid_dataset = SpotifyDataset(test_tracks, test_skips, track_vocab, bert=True, bert_mask_proportion = 0.2)

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn =bert_collate_fn)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn =bert_collate_fn)


learning_rate = 1e-4
model = Transformer(INPUT_SIZE, OUTPUT_SIZE, device, max_length = MAX_LEN)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch_idx in range(EPOCHS):
    print("-----------------------------------")
    print("Epoch %d" % (epoch_idx+1))
    print("-----------------------------------")
    
    train_loss, avg_train_loss = train(model, train_loader, optimizer, criterion)
    scheduler.step(train_loss)

    val_loss, avg_val_loss = evaluate(model, valid_loader, criterion)

    avg_train_loss = avg_train_loss.item()
    avg_val_loss = avg_val_loss.item()
    print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))
