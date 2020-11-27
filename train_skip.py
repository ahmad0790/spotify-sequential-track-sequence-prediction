import math
import time
import numpy as np
import pandas as pd
import glob
import pickle
import random
import tqdm
import os

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.SkipTransformer import StandardTransformer
from datasets.SpotifyDataset import SpotifyDataset, bert_collate_fn, custom_collate_fn
from torch.nn import functional as F

#INIT PARAMS
PATH_OUTPUT = "output/"
model_name = 'model_base_transformer_skip_embed_v2'
torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#FROM BD4H CLASS
def plot_learning_curves(model_name, train_losses, valid_losses, train_accuracies, valid_accuracies):

    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.savefig('output/' + model_name + '_loss_curve.png')

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.savefig('output/' + model_name + '_accuracy_curve.png')


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

def train_transformer_model(model, dataloader, optimizer, criterion, scheduler = None, device = None, skip=False):

    print("SKIP FLAG")
    print(skip)
    model.train()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(dataloader):

        input_sequence = input_sequence.to(device)
        input_sequence[input_skips==1] = 1

        if skip==False:
            label = label_sequence.cuda()
            outputs = model(input_sequence, label)
        else:
            label_sequence = label_sequence.cuda()
            label = label_skips.cuda()
            input_skips = input_skips.cuda()
            outputs = model(input_sequence, label_sequence, input_skips)

        acc = mean_average_accuracy(outputs, label)

        if batch_idx %100 == 0:
            print("PREDICTED SKIP SEQUENCE")
            print(torch.argmax(outputs, dim=2)[0,:])
            print("ACTUAL SKIP SEQUENCE")
            print(label[0,:])

        outputs = outputs.reshape(-1, outputs.shape[-1])
        label = label.reshape(-1)
        optimizer.zero_grad()
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        avg_loss.update(loss.item(), label.size(0))
        avg_acc.update(acc.item(), label.size(0))

        if batch_idx % 100 ==0:
            print("Batch: %d, Train Loss: %.4f, Train Accuracy: %.4f" % ((batch_idx+1), avg_loss.avg, avg_acc.avg))

        '''
        if (batch_idx+1) % 200 ==0:
            break
        '''

    return avg_loss.avg, avg_acc.avg

def evaluate_transformer_model(model, dataloader, optimizer, criterion, scheduler = None, device = None, skip=False):

    print("EVALUATION")

    model.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    with torch.no_grad():

        for batch_idx, (input_sequence, input_skips, label_sequence, label_skips) in enumerate(dataloader):

            input_sequence = input_sequence.to(device)

            if skip==False:
                label = label_sequence.cuda()
                outputs = model(input_sequence, label)
            else:
                label_sequence = label_sequence.cuda()
                label = label_skips.cuda()
                input_skips = input_skips.cuda()
                outputs = model(input_sequence, label_sequence, input_skips)

            acc = mean_average_accuracy(outputs, label)

            if batch_idx %100 == 0:
                print("PREDICTED SKIP SEQUENCE")
                print(torch.argmax(outputs, dim=2)[0,:])
                print("ACTUAL SKIP SEQUENCE")
                print(label[0,:])

            outputs = outputs.reshape(-1, outputs.shape[-1])
            label = label.reshape(-1)

            loss = criterion(outputs, label)

            avg_loss.update(loss.item(), label.size(0))
            avg_acc.update(acc.item(), label.size(0))

            '''
            if (batch_idx+1) % 200 ==0:
                print("Batch: %d, Val Loss: %.4f, Val Accuracy: %.4f" % ((batch_idx+1), avg_loss.avg, avg_acc.avg))
                break
            '''

    return avg_loss.avg, avg_acc.avg


#Read the data
print("READING THE DATA")

with open("data/all_session_tracks_train.pkl", 'rb') as f:
    train_tracks = pickle.load(f)

with open("data/all_session_skips_train.pkl", 'rb') as f:
    train_skips = pickle.load(f)

with open("data/all_session_tracks_test.pkl", 'rb') as f:
    test_tracks = pickle.load(f)
    test_tracks = test_tracks[0:100000]

with open("data/all_session_skips_test.pkl", 'rb') as f:
    test_skips = pickle.load(f)
    test_skips = test_skips[0:100000]

with open("data/track_vocabs.pkl", 'rb') as f:
    track_vocab = pickle.load(f)

track_feats = np.load('data/track_embedding.npy')

print("VOCAB SIZE")
print(len(track_vocab))

print("TRAIN SESSIONS SIZE")
print(len(train_tracks))

print("VALID SESSIONS SIZE")
print(len(test_tracks))

INPUT_SIZE = len(track_vocab)
OUTPUT_SIZE = len(track_vocab)
PAD_IDX = track_vocab['pad']
BATCH_SIZE = 128
MAX_LEN = 20
EPOCHS = 5
SKIP = True
PAD_MASK = 1

train_dataset = SpotifyDataset(train_tracks, train_skips, track_vocab, bert=False, bert_mask_proportion = 0.2, skip_pred=SKIP, padding=False)
valid_dataset = SpotifyDataset(test_tracks, test_skips, track_vocab, bert=False, bert_mask_proportion = 0.2, skip_pred=SKIP, padding=False)

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn =custom_collate_fn)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn =custom_collate_fn)

#OPTIM PARAMETERS
learning_rate = 1e-4

model=StandardTransformer(vocab_size =INPUT_SIZE, d_model=128, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, max_seq_length=10, skip_pred = SKIP, feat_embed = track_feats, device = device, padding=False)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

if SKIP:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

best_val_loss = 100000000
for epoch_idx in range(EPOCHS):
    print("-----------------------------------")
    print("Epoch %d" % (epoch_idx+1))
    print("-----------------------------------")
    
    avg_train_loss, avg_train_acc = train_transformer_model(model, train_loader, optimizer, criterion, scheduler = None, device = device, skip=SKIP)
    avg_val_loss, avg_val_acc = evaluate_transformer_model(model, valid_loader, optimizer, criterion, scheduler = None, device = device, skip=SKIP)
    #scheduler.step(train_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

    print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("Training Accuracy: %.4f. Validation Accuracy: %.4f. " % (avg_train_acc, avg_val_acc))
    print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))

    if avg_val_loss < best_val_loss:
        print('Found Best')
        best_val_loss = avg_val_loss
        torch.save(model, os.path.join(PATH_OUTPUT, model_name + '.pth'))
        print('Saved Best Model')

##SAVING OUTPUT AND LEARNING CURVES
plot_learning_curves(model_name, train_losses, val_losses, train_accs, val_accs)

epoch_range = list(range(0, EPOCHS))
train_stats = pd.DataFrame(list(zip(epoch_range, train_accs, val_accs, train_losses, val_losses)), 
               columns =['Epoch', 'Train Accuracy', 'Valid Accuracy', 'Train Loss', 'Valid Loss'])
print('TRAINING/VALIDATION SUMMARY')
print(train_stats.head(len(epoch_range)))
print('')
train_stats.to_csv ('output/'+model_name+'.csv', index = None, header=True)
print("DONE EVALUATING")
