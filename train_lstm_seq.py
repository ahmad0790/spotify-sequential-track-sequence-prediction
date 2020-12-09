import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
import pickle
import torch
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import torch.nn.functional as F

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("READING DATA")
train_tracks = pickle.load(open('data/all_session_tracks_train.pkl','rb'))
test_tracks = pickle.load(open('data/all_session_tracks_test.pkl','rb'))
train_skips = pickle.load(open('data/all_session_skips_train.pkl','rb'))
test_skips = pickle.load(open('data/all_session_skips_test.pkl','rb'))
track_vocab = pickle.load(open('data/track_vocabs.pkl','rb'))

split_index = int(0.8*len(train_tracks))
train_tracks, valid_tracks, train_skips, valid_skips =  train_tracks[0:split_index],train_tracks[split_index:],train_skips[0:split_index],train_skips[split_index:]
train_tracks, train_skips = train_tracks, train_skips
#train_tracks, train_skips = train_tracks[:1000], train_skips[:1000]

valid_tracks, valid_skips = valid_tracks[:1000], valid_skips[:1000]
test_tracks, test_skips = test_tracks[:100000], test_skips[:100000]

track_embeddings = np.load(open('data/track_embedding.npy','rb')).astype('double')
track_embeddings = torch.from_numpy(normalize(track_embeddings,axis=0)).double()


# ## Vanilla Seq2Seq Model Architecture
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target):

    """Computes the accuracy of the predicted skip sequence"""

    seq_len = target.shape[1]
    correct = output.eq(target)
    correct = correct.sum(axis=1) * 1.0
    acc = correct / seq_len
    return acc
def mean_average_accuracy(output, target):

    """Computes the  mean average accuracy of the predicted skip sequence up till the 10th seq position"""
    output = output.to(device)
    target = target.to(device)
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

    """Computes the  mean average accuracy of the predicted skip sequence up till the 10th seq position"""
    output = output.to(device)
    target = target.to(device)

    T = output.shape[1]
    batch_size = target.shape[0]
    output = torch.argmax(output, dim=2)
    A_i = torch.zeros(batch_size, T)
    L_i = torch.zeros(batch_size, T)
    acc = torch.zeros(T)

    for i in range(T):
        acc[i] = torch.mean(accuracy(output[:,i].reshape(batch_size,1), target[:,i].reshape(batch_size,1)))

    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#MODEL ITSELF
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding.from_pretrained(track_embeddings)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        embedded = self.dropout(self.embedding(src)).float()
        #embedded = F.relu(embedded)
        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding.from_pretrained(track_embeddings)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=False)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input)).float()

        #embedded = [1, batch size, emb dim]
        #embedded = F.relu(embedded)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        #prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq1(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim,             "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,             "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        src = src.cuda()
        trg = trg.cuda()

        split_index = math.floor(src.shape[0]//2)

        seq_to_encode = src[0:split_index,:]
        seq_to_decode = src[split_index:,:]
        targets_of_decoded = trg

        batch_size = trg.shape[1]
        trg_len = targets_of_decoded.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(seq_to_encode)

        #first input to the decoder is the <sos> tokens
        #input = trg[split_index-1,:] #CHANGE1
        
        if teacher_forcing_ratio==0:
            input = seq_to_encode[-1,:]
        else:
            input = seq_to_decode[0,:]

        for t in range(trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            if teacher_forcing_ratio == 0:
                #get the highest predicted token from our predictions
                top1 = output.argmax(1)

                #if teacher forcing, use actual next token as next input
                #if not, use predicted token
                input = top1

            if teacher_forcing_ratio == 1 and (t < trg_len-1):
                #input =  trg[t,:]
                input =  seq_to_decode[t+1,:]

        return outputs

class DatasetSpotify(Dataset):

    def __init__(self, tracks, skips, transform=None):
        self.tracks = tracks
        self.skips = skips
        self.transform = transform

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        features = self.tracks[index]
        label = self.skips[index]

        return np.array(features), np.array(features)

##TRAINING AND EVALUATION
def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0
    total_maa = 0

    for i, (x,y) in enumerate(iterator):

        src = x.permute(1,0)
        trg = y.permute(1,0)
        
        src = src.cuda()
        trg = trg.cuda()

        optimizer.zero_grad()
        split_index = math.floor(src.shape[0]//2)
        trg = trg[split_index:,:]

        output = model(src, trg, 1)
        maa = mean_average_accuracy(output.permute(1,0,2),trg.permute(1,0))

        if i % 100 == 0:
            print('Loss',i,epoch_loss/i)
            print('mean avg accuracy', mean_average_accuracy(output.permute(1,0,2)[10:],trg.permute(1,0)[10:]))
            #print('accuracy_at_k', accuracy_at_k(output.permute(1,0,2)[10:],trg.permute(1,0)[10:]))

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        #print('here initial output shape is ',output.shape)

        output = output[1:].view(-1, output_dim)
        #print('output shapes are',output.shape, trg.shape)

        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        total_maa += maa

    return epoch_loss / len(iterator), total_maa / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    total_maa = 0
    total_acc = torch.zeros(10)

    with torch.no_grad():

        for i, (x,y) in enumerate(iterator):

            src = x.permute(1,0)
            trg = y.permute(1,0)

            src = src.cuda()
            trg = trg.cuda()

            split_index = math.floor(src.shape[0]//2)

            trg = trg[split_index:,:]
            output = model(src, trg, 0)

            max_output = torch.argmax(output,dim=2)
            maa = mean_average_accuracy(output.permute(1,0,2),trg.permute(1,0))
            local_acc = accuracy_at_k(output.permute(1,0,2),trg.permute(1,0))

            #print(output.shape, max_output.permute(1,0).reshape(-1,20).shape,trg.permute(1,0).reshape(-1,20).shape)
            if i % 1000 == 0:
                maa = mean_average_accuracy(output.permute(1,0,2)[10:],trg.permute(1,0)[10:])
                print('iter ',i)
                print(max_output[:10][10:])
                print('acc is', torch.mean(accuracy(max_output.permute(1,0).reshape(-1,20),trg.permute(1,0).reshape(-1,20))))
                print('mean avg accuracy', maa)
                print('accuracy_at_k', accuracy_at_k(output.permute(1,0,2)[10:],trg.permute(1,0)[10:]))

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            total_maa += maa
            total_acc += local_acc

    return epoch_loss / len(iterator), total_maa / len(iterator) , total_acc/len(iterator)


###TRAINING#####
print("TRAINING")
BATCH_SIZE = 256
INPUT_DIM = len(track_vocab)
OUTPUT_DIM = len(track_vocab)

ENC_EMB_DIM = track_embeddings.shape[1]
DEC_EMB_DIM = track_embeddings.shape[1]
HID_DIM = 512
N_LAYERS = 1
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
N_EPOCHS = 2
CLIP = 1

train_dataset = DatasetSpotify(train_tracks,train_skips)
valid_dataset = DatasetSpotify(valid_tracks,valid_skips)
test_dataset = DatasetSpotify(test_tracks,test_skips)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

train_iter, valid_iter, test_iter = iter(train_loader),iter(valid_loader),iter(test_loader)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq1(enc, dec, device).to(device)
#model.apply(init_weights)
model.load_state_dict(torch.load('vanilla-sequence-new.pt'))

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
best_valid_loss = float('inf')
best_valid_maa = 0.0000
epoch_accs = []
for epoch in range(N_EPOCHS):
    train_iter, valid_iter, test_iter = iter(train_loader),iter(valid_loader),iter(test_loader)

    start_time = time.time()

    train_loss, train_maa =  train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss, valid_maa, _ = evaluate(model, test_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if (valid_loss < best_valid_loss) or (valid_maa > best_valid_maa):
        best_valid_loss = valid_loss
        best_valid_maa = valid_maa
        torch.save(model.state_dict(), 'vanilla-sequence-new-v2.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train MAA: {train_maa:7.3f}' )
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Valid MAA: {valid_maa:7.3f}')
    epoch_accs.append((train_maa, valid_maa))
    print(epoch_accs)

print(epoch_accs)

BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_iter, valid_iter, test_iter = iter(train_loader),iter(valid_loader),iter(test_loader)

model.load_state_dict(torch.load('vanilla-sequence-new-v2.pt'))

test_loss, test_maa, acc_at_k = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test MAA: {test_maa:7.3f}')
print('acc at k', acc_at_k)