import math
import copy
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class BertModelDataset(Dataset):
    
    def __init__(self, tracks, skips, track_vocab=None, bert_mask_proportion=0.2, skip=False):
        
        if len(tracks) != len(skips):
            raise ValueError("Session Tracks and Session Skips have different sizes")
        
        self.tracks = tracks
        self.skip_tracks = skips
        self.skip = skip
        self.PAD_INDEX = track_vocab['pad']
        self.MASK_INDEX = track_vocab['mask']
        self.bert_mask_proportion = bert_mask_proportion
        self.len_track_vocab = len(track_vocab)
        self.SESSION_HALF_LENGTH = 10

        print("BERT MASK PROB")
        print(self.bert_mask_proportion)

    def __len__(self):
        return len(self.tracks)
    
    # return a list of length 20 the size of each session
    def __getitem__(self, index):

        # for BERT pretraining we only need to train on the masked train sequence
        if self.skip==False:
            masked_sequence, label_sequence = self.mask_words(self.tracks[index])
            
            return masked_sequence, label_sequence
        else:
            #for regular BERT we just get the usual input
            return self.tracks[index], self.skip_tracks[index]


    #only for BERT masking words for fine tuning after pre training
    def mask_words(self, input_sequence):

        '''
        labels = copy.deepcopy(input_sequence[self.SESSION_HALF_LENGTH:])
        masked_labels = [self.PAD_INDEX for i in range(self.SESSION_HALF_LENGTH)]
        masked_labels.extend(labels)

        masked_sequence = copy.deepcopy(input_sequence[0:self.SESSION_HALF_LENGTH])
        masked_tokens = [self.MASK_INDEX for i in range(self.SESSION_HALF_LENGTH)]
        masked_sequence.extend(masked_tokens)
        '''
        masked_sequence = input_sequence[0:19]
        masked_labels = input_sequence[1:20]
        
        return masked_sequence, masked_labels

def bert_collate_fn(batch):

    input_sequence = torch.LongTensor([item[0] for item in batch])
    label_sequence = torch.LongTensor([item[1] for item in batch])

    return input_sequence, label_sequence
