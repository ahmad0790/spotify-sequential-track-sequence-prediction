import numpy as np
import random
import math 
import torch
from einops import rearrange
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F

#REFERENCE: https://andrewpeng.dev/transformer-pytorch/
#https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
def gen_nopeek_mask(length):
	mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

class StandardTransformer(nn.Module):

	def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, skip_pred = False, feat_embed = None, device =None, padding=False, track_feat_embed = None):
		super().__init__()

		self.PAD_MASK = 0
		self.device = device
		self.padding = padding
		self.skip_pred = skip_pred
		print("SKIP")
		print(self.skip_pred)

		if feat_embed is not None:
			self.feat_embed = True
		else:
			self.feat_embed = False

		self.num_track_feats = 26
		self.vocab_size = vocab_size
		self.d_model = d_model #the number of expected features in the encoder/decoder inputs (default=512)
		self.max_length = max_seq_length
		self.dim_feedforward = dim_feedforward
		self.nhead = nhead
		self.num_decoder_layers = num_decoder_layers
		self.num_encoder_layers = num_encoder_layers

		self.skip_embed = nn.Embedding(2, self.d_model+self.num_track_feats)
		self.feat_embed = nn.Embedding(self.vocab_size, self.num_track_feats)
		self.track_feat_embed = nn.Embedding(self.vocab_size, self.d_model)
		self.pos_embed = nn.Embedding(self.max_length, self.d_model+self.num_track_feats)
		self.d_model_combined = self.d_model+self.num_track_feats

		self.track_feat_embed.weight.data.copy_(torch.from_numpy(track_feat_embed))

		#track_feat_weights = torch.FloatTensor(track_feat_embed).to(self.device)			
		#self.track_feat_embed.weights = nn.Parameter(track_feat_weights, requires_grad=True)
		#print(self.track_feat_embed.weights.shape)

		if feat_embed is not None:
			feat_weights = torch.FloatTensor(feat_embed).to(self.device)
			feat_weights[1,:] = torch.rand(self.num_track_feats)
			feat_weights = F.normalize(feat_weights, p=2, dim=1)
			self.feat_embed.weights = nn.Parameter(feat_weights, requires_grad=True)

		self.transformer = nn.Transformer(d_model=self.d_model_combined, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers)
		
		#change dimension size to 2 if predicting Skip
		if self.skip_pred==False:
			self.fc = nn.Linear(self.d_model_combined, self.vocab_size)
		else:
			self.fc = nn.Linear(self.d_model_combined, 2)

	
	def forward(self, src, tgt, skip_sequence, verbose=False):

		#mask for padded words in source
		if self.padding:
			#mask for padded words in source
			src_key_padding_mask = torch.zeros(src.size()).bool().to(self.device)
			src_key_padding_mask[src==self.PAD_MASK] = True

			#mask for padded words in target
			tgt_key_padding_mask = torch.zeros(tgt.size()).bool().to(self.device)
			tgt_key_padding_mask[tgt==self.PAD_MASK] = True

			#encoder values that are masked because of padding. same as src padding mask
			memory_key_padding_mask = src_key_padding_mask.clone()

		#get positional embeddings (#num rows in batch x seq len)
		positional_encoding = torch.zeros(src.shape[0], src.shape[1]).to(torch.int64)
		positional_encoding = positional_encoding.to(self.device)
		for i in range(src.shape[0]):
			positional_encoding[i,:] = torch.LongTensor([list(range(0,src.shape[1]))])
		
		positional_embeddings = self.pos_embed(positional_encoding)
	
		#right shift target embedding by 1 (last token is not predicted)
		if self.skip_pred == False:
			tgt = torch.cat((src[:,-1].reshape(src.shape[0],1), tgt[:, :-1]), 1)
			source_seq_embeddings = torch.cat((self.track_feat_embed(src), self.feat_embed(src)), dim=2) + positional_embeddings
			target_sequence_embeddings = torch.cat((self.track_feat_embed(tgt), self.feat_embed(tgt)), dim=2) + positional_embeddings 

		else:
			source_seq_embeddings = torch.cat((self.track_feat_embed(src), self.feat_embed(src)), dim=2) + positional_embeddings
			#source_seq_embeddings = torch.cat((self.track_feat_embed(src), self.feat_embed(src)), dim=2) + positional_embeddings + self.skip_embed(skip_sequence)
			target_sequence_embeddings = torch.cat((self.track_feat_embed(tgt), self.feat_embed(tgt)), dim=2) + positional_embeddings 

		#generate no peek look ahead mask for target sequence
		tgt_no_peek_mask = gen_nopeek_mask(tgt.shape[1]).to(self.device)
		source_seq_embeddings = rearrange(source_seq_embeddings, 'n s t -> s n t')
		target_sequence_embeddings = rearrange(target_sequence_embeddings, 'n s t -> s n t')

		if self.padding:
			output = self.transformer(
									src = source_seq_embeddings, 
									tgt = target_sequence_embeddings, 
									tgt_mask=tgt_no_peek_mask, 
									src_key_padding_mask=src_key_padding_mask,
									tgt_key_padding_mask=tgt_key_padding_mask, 
									memory_key_padding_mask=memory_key_padding_mask
									)
		else:
			output = self.transformer(
						src = source_seq_embeddings, 
						tgt = target_sequence_embeddings, 
						tgt_mask=tgt_no_peek_mask
						)

		output = rearrange(output, 't n e -> n t e')		
		output = self.fc(output)

		return output
