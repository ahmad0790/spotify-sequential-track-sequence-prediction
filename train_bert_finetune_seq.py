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
from models.BertTransformer import BertTransformer
from datasets.BertModelDataset import BertModelDataset, bert_collate_fn
from torch.nn import functional as F

#INIT PARAMS
EPOCHS = 5
model = torch.load('data/model_base_bert_0.2_1e-4.pth', map_location=torch.device('cpu'))
print("LOADED BERT MODEL")

PATH_OUTPUT = "output/"
model_name = 'model_bert_finetune_seq_v2'
torch.manual_seed(98675476)
print(model_name)

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

def accuracy_raw(output, target):
    
    """Computes the average accuracy of the predicted skip sequence"""
    
    output = torch.argmax(output, dim=2)
    output = output[target>0]
    target = target[target>0]
    correct = output.eq(target)
    correct = correct.sum() * 1.0
    acc = correct / target.shape[0]
    return acc

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


##FOR BERT LM MASKED PRETRAINING/FINETUNING
def train_bert(model, dataloader, optimizer, criterion, scheduler = None, device=None, epoch_idx=None):

    model.train()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for batch_idx, (masked_sequence, label_sequence) in enumerate(dataloader):

        input_sequence = masked_sequence.to(device)
        #input_sequence = masked_sequence[:,0:19].to(device)
        label = label_sequence.cuda()
        #label = label_sequence[:,1:].cuda()


        outputs = model(input_sequence)
        acc = mean_average_accuracy(outputs[:,10:,:], label[:,10:])

        if batch_idx %1000 == 0:
            print("MASKED SEQUENCE")
            print(input_sequence[0,:])
            print("PREDICTED SEQUENCE")
            print(torch.argmax(outputs, dim=2)[0,:])
            print("LABEL SEQUENCE")
            print(label[0,:])

        outputs = outputs.reshape(-1, outputs.shape[-1])
        label = label.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(outputs, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        avg_loss.update(loss.item(), label.size(0))
        avg_acc.update(acc.item(), label.size(0))

        if batch_idx % 1000 ==0:
            print("Epoch: %d, Batch: %d, Train Loss: %.4f, Train Accuracy: %.4f" % (epoch_idx, (batch_idx+1), avg_loss.avg, avg_acc.avg))

        '''
        if (batch_idx+1) % 100 ==0:
            break
        '''

    return avg_loss.avg, avg_acc.avg


def evaluate_bert(model, dataloader, optimizer, criterion, scheduler = None, device = None, skip=False, epoch_idx=None):

    model.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for batch_idx, (masked_sequence, label_sequence) in enumerate(dataloader):

        input_sequence = masked_sequence.to(device)
        label = label_sequence.cuda()

        outputs = model(input_sequence)
        acc = mean_average_accuracy(outputs[:,10:20,:], label[:,10:20])

        if batch_idx %100 == 0:
            print("MASKED SEQUENCE")
            print(input_sequence[0,:])
            print("PREDICTED SEQUENCE")
            print(torch.argmax(outputs, dim=2)[0,:])
            print("LABEL SEQUENCE")
            print(label[0,:])

        outputs = outputs.reshape(-1, outputs.shape[-1])
        label = label.reshape(-1)

        loss = criterion(outputs, label)

        avg_loss.update(loss.item(), label.size(0))
        avg_acc.update(acc.item(), label.size(0))

        if batch_idx % 1000 ==0:
            print("Epoch: %d, Batch: %d, Train Loss: %.4f, Train Accuracy: %.4f" % (epoch_idx, (batch_idx+1), avg_loss.avg, avg_acc.avg))

        '''
        if (batch_idx+1) % 200 ==0:
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
PAD_IDX = 0
BATCH_SIZE = 128
MAX_LEN = 20
SKIP = False
bert_masking_prob=0.2

train_dataset = BertModelDataset(train_tracks, train_skips, track_vocab, bert_mask_proportion = bert_masking_prob, skip=SKIP)
valid_dataset = BertModelDataset(test_tracks, test_skips, track_vocab, bert_mask_proportion = bert_masking_prob, skip=SKIP)

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn =bert_collate_fn)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn =bert_collate_fn)

#OPTIM PARAMETERS
learning_rate = 1e-4
print(learning_rate)

#model=BertTransformer(vocab_size =INPUT_SIZE, d_model=128, nhead=2, num_encoder_layers=2, dim_feedforward=2048, max_seq_length=20, device=device, skip_finetune=SKIP, pretrained_embed=track_feats)
model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
train_losses, val_losses, train_accs, val_accs = [], [], [], []

best_val_loss = 100000000
best_val_acc = 0.00000
for epoch_idx in range(EPOCHS):
    print("-----------------------------------")
    print("Epoch %d" % (epoch_idx+1))
    print("-----------------------------------")
    
    avg_train_loss, avg_train_acc = train_bert(model, train_loader, optimizer, criterion, scheduler = None, device = device, epoch_idx=epoch_idx)
    avg_val_loss, avg_val_acc = evaluate_bert(model, valid_loader, optimizer, criterion, scheduler = None, device = device, epoch_idx=epoch_idx)
    #scheduler.step(train_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

    print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("Training Accuracy: %.4f. Validation Accuracy: %.4f. " % (avg_train_acc, avg_val_acc))
    print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))

    if (avg_val_loss < best_val_loss) or (avg_val_acc > best_val_acc):
        print('Found Best')
        best_val_loss = avg_val_loss
        best_val_acc = avg_val_acc
        torch.save(model, os.path.join(PATH_OUTPUT, model_name + '.pth'))
        torch.save(model.state_dict(), os.path.join(PATH_OUTPUT, model_name + '_dict' + '.pth'))
        print('Saved Best Model')
        #bert_embedding_weights = model.embed_src.weight.detach().cpu().numpy()
        #pd.DataFrame(bert_embedding_weights).to_csv('output/bert_emb_'+model_name+'.csv')
        #print("SAVED EMBEDDING")

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

#SAVE LEARNED EMBEDDINGS
bert_embedding_weights = model.embed_src.weight.detach().cpu().numpy()
pd.DataFrame(bert_embedding_weights).to_csv('output/bert_emb_'+model_name+'final'+'.csv')
print("SAVED EMBEDDING")

