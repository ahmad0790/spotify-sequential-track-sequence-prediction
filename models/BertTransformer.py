import numpy as np
import random
import math 
import torch
from einops import rearrange
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F


class BertTransformer(nn.Module):

	def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length, device=None):
		super().__init__()

		self.device = device
		self.PAD_MASK = 0
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.max_length = max_seq_length
		self.dim_feedforward = dim_feedforward
		self.nhead = nhead
		self.num_encoder_layers = num_encoder_layers
		self.embed_src = nn.Embedding(self.vocab_size, d_model)
		self.pos_enc = nn.Embedding(self.max_length, d_model)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)
		self.fc = nn.Linear(self.d_model, self.vocab_size)

	def forward(self, src):

		batch_size = src.shape[0]
		seq_len = src.shape[1]

		#print(seq_len)

		#src_key_padding_mask = torch.zeros(src.size()).bool()
		#src_key_padding_mask[src==self.PAD_MASK] = True

		input_embeddings = self.embed_src(src)

		#get positional embeddings
		positional_encoding = torch.zeros(batch_size, seq_len).to(torch.int64)
		positional_encoding = positional_encoding.to(self.device)

		for i in range(batch_size):
			positional_encoding[i,:] = torch.LongTensor([list(range(0, seq_len))])
		positional_embeddings = self.pos_enc(positional_encoding)

		source_seq_embeddings = input_embeddings + positional_embeddings
		source_seq_embeddings = rearrange(source_seq_embeddings, 'n s t -> s n t')

		output = self.transformer_encoder(source_seq_embeddings)
		
		output = rearrange(output, 't n e -> n t e')		
		output = self.fc(output)

		return output
		