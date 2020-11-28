import math
import copy
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class BertModelDataset(Dataset):
    
    def __init__(self, tracks, skips, track_vocab=None, skip=False, bert_mask_proportion=0.2):
        
        if len(tracks) != len(skips):
            raise ValueError("Session Tracks and Session Skips have different sizes")
        
        self.tracks = tracks
        self.skips = skips
        self.PAD_INDEX = track_vocab['pad']
        self.MASK_INDEX = track_vocab['mask']
        self.bert_mask_proportion = bert_mask_proportion
        self.len_track_vocab = len(track_vocab)
        self.self.SESSION_HALF_LENGTH = 10

        print("BERT MASK PROB")
        print(self.bert_mask_proportion)

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):

        # for BERT pretraining we only need to train on the masked train sequence
        if self.skip==False:

            masked_sequence, label_sequence = self.mask_words(self.tracks[index])
            
            return masked_sequence, label_sequence
        else:
            return self.tracks[index], self.skips[index]


    #only for BERT masking words for fine tuning after pre training
    def mask_words(self, input_sequence):

        labels = copy.deepcopy(input_sequence)
        masked_sequence = copy.deepcopy(input_sequence[0:self.SESSION_HALF_LENGTH])
        masked_tokens = [self.MASK_INDEX for i in range(self.SESSION_HALF_LENGTH)]
        masked_sequence.extend(masked_tokens)
        
        return masked_sequence, labels

def bert_collate_fn(batch):

    input_sequence = torch.LongTensor([item[0] for item in batch])
    label_sequence = torch.LongTensor([item[1] for item in batch])

    return input_sequence, label_sequence
