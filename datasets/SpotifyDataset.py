import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class SpotifyDataset(Dataset):
    
    def __init__(self, tracks, skips, track_vocab=None, bert=False, bert_mask_proportion=0.2, skip_pred = False, padding=False):
        
        if len(tracks) != len(skips):
            raise ValueError("Session Tracks and Session Skips have different sizes")
        
        self.tracks = tracks
        self.skips = skips
        self.PAD_INDEX = track_vocab['pad']

        #self.MASK_INDEX = 0
        #self.MASK_INDEX  = 1
        self.MASK_INDEX = track_vocab['mask']
        
        self.SESSION_HALF_LENGTH = 10
        self.bert = bert
        self.bert_mask_proportion = bert_mask_proportion
        self.len_track_vocab = len(track_vocab)
        self.padding = padding
        self.skip_pred = skip_pred

        print("BERT MASK PROB")
        print(self.bert_mask_proportion)

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):

        # for BERT pretraining we only need to train on the masked train sequence
        if self.bert==True:

            masked_sequence, label_sequence = self.mask_words(self.tracks[index])
            
            #pad to make all sessions even length
            if self.padding:
                padding_tokens = [self.PAD_INDEX for i in range(self.SESSION_HALF_LENGTH - len(label_sequence))]
                masked_sequence.extend(padding_tokens)
                label_sequence.extend(padding_tokens)

            return masked_sequence, label_sequence

        else:

            input_data, labels_data = self.split_session(self.tracks[index], self.skips[index])

            input_sequence, input_skips = input_data
            label_sequence, label_skips = labels_data
            seq_len = len(input_sequence)

            if self.padding:
                #pad to make all sessions even length
                padding_tokens = [self.PAD_INDEX for i in range(self.SESSION_HALF_LENGTH - len(input_sequence))]
                input_sequence.extend(padding_tokens)
                input_skips.extend(padding_tokens)
               
                padding_tokens = [self.PAD_INDEX for i in range(self.SESSION_HALF_LENGTH - len(label_sequence))]
                label_sequence.extend(padding_tokens)
                label_skips.extend(padding_tokens)

            return input_sequence, input_skips, label_sequence, label_skips

    def split_session(self, sequence, skip_sequence):

        seq_len = math.floor(len(sequence)//2)

        train_sequence = sequence[0:seq_len]
        test_sequence = sequence[seq_len:]
        train_skips = skip_sequence[0:seq_len]
        test_skips = skip_sequence[seq_len:]

        input_data = (train_sequence, train_skips)
        labels_data = (test_sequence, test_skips)

        return input_data, labels_data

    #only for BERT masking words for pre training
    def mask_words(self, train_sequence):

        seq_len = len(train_sequence)
        labels = []

        for i in range(seq_len):

            mask_prob = random.random()
            track_index = train_sequence[i]

            if mask_prob <= self.bert_mask_proportion:
                
                #mask_prob = random.random()
                mask_prob /= self.bert_mask_proportion

                if mask_prob <= 0.8:
                    train_sequence[i] = self.MASK_INDEX

                elif mask_prob <= 0.9:
                    train_sequence[i] = random.randrange(self.len_track_vocab)

                else:
                    train_sequence[i] = track_index

                labels.append(track_index)

            else:

                #set to 0 any word that is not masked. These will not be trained on in BERT
                labels.append(0)

        return train_sequence, labels

def custom_collate_fn(batch):
    input_sequence = torch.LongTensor([item[0] for item in batch])
    input_skips = torch.LongTensor([item[1] for item in batch])
    label_sequence = torch.LongTensor([item[2] for item in batch])
    label_skips = torch.LongTensor([item[3] for item in batch])

    return input_sequence, input_skips, label_sequence, label_skips

def bert_collate_fn(batch):

    input_sequence = torch.LongTensor([item[0] for item in batch])
    label_sequence = torch.LongTensor([item[1] for item in batch])

    return input_sequence, label_sequence
