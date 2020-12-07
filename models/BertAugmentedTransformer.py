import numpy as np
import random
import math 
import torch
from einops import rearrange
from torch import nn
from models.CustomizedTransformer import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F

class BertAugmentedTransformer(nn.Module):

	def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, skip_pred = False, feat_embed=None, device=None):
		super(BertAugmentedTransformer, self).__init__()

		self.device = device
		self.max_length = max_seq_length
		self.d_model = d_model
		self.vocab_size = vocab_size
		self.nhead = nhead	
		self.num_decoder_layers = num_decoder_layers
		self.num_encoder_layers = num_encoder_layers
		self.num_track_feats = 26
		self.d_model_combined = self.d_model+self.num_track_feats

		self.layer_norm = nn.LayerNorm(self.d_model_combined)

		self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model_combined, nhead=self.nhead, dropout=0.2, activation = 'relu')
		self.transformer_encoder = TransformerEncoder(self.encoder_layer, self.num_encoder_layers, norm= self.layer_norm)
		
		self.decoder_layer = TransformerDecoderLayer(d_model=self.d_model_combined, nhead=self.nhead, dropout = 0.2, activation= 'relu')
		self.transformer_decoder = TransformerDecoder(self.decoder_layer, self.num_decoder_layers, norm = self.layer_norm)

		self.skip_embed = nn.Embedding(2, self.d_model_combined)
		self.feat_embed = nn.Embedding(self.vocab_size, self.num_track_feats)
		self.track_embed = nn.Embedding(self.vocab_size, self.d_model_combined)
		self.pos_embed = nn.Embedding(self.max_length, self.d_model_combined)

		feat_weights = torch.FloatTensor(feat_embed).to(self.device)
		feat_weights[1,:] = torch.rand(self.num_track_feats)
		feat_weights = F.normalize(feat_weights, p=2, dim=1)
		self.feat_embed.weights = nn.Parameter(feat_weights, requires_grad=True)

		if skip_pred:
			self.fc = nn.Linear(self.d_model_combined, 2)
		else:
			self.fc = nn.Linear(self.d_model_combined, self.vocab_size)

	#src bert and tgt bert are the bert fina layere output. they are the same.
	def forward(self, src, tgt, bert_output, src_skip_inputs=None, src_mask=None, tgt_mask=None, encoder_mask=None):
		"Take in and process masked src and target sequences."

		#only during training for track sequences we do teacher forcing
		#teacher forcing right shift target embedding by 1 (last token is not predicted)
		if src_skip_inputs is None:
			tgt = torch.cat((src[:,-1].reshape(src.shape[0],1), tgt[:, :-1]), 1)

		encoder_output = self.encode(src, bert_output)
		decoder_output = self.decode(encoder_output, tgt, bert_output)
		output = self.fc(decoder_output)
		output = rearrange(output, 't n e -> n t e')
		return output
	
	def encode(self, src, src_bert, src_skip_inputs=None, src_mask=None):
		source_embeddings = self.embed(src, src_skip_inputs)
		source_embeddings = rearrange(source_embeddings, 'n s t -> s n t')
		x = self.transformer_encoder(source_embeddings, src_bert)
		return x
	
	def decode(self, encoder_output, tgt, tgt_bert, src_mask=None, tgt_mask=None, encoder_mask = None):
		tgt_embeddings = self.embed(tgt)
		tgt_no_peek_mask = self.gen_nopeek_mask(tgt.shape[1]).to(self.device)
		tgt_embeddings = rearrange(tgt_embeddings, 'n s t -> s n t')
		x = self.transformer_decoder(tgt = tgt_embeddings, tgt_bert = tgt_bert, memory = encoder_output, tgt_mask=tgt_no_peek_mask)
		return x

	def embed(self, src, src_skip_inputs=None):
		"""
		:param inputs: intTensor of shape (N,T)
		:returns embeddings: floatTensor of shape (N,T,H)
		"""

		track_id_embed = self.track_embed(src)
		track_feat_embed = self.feat_embed(src)

		positional_encoding = torch.zeros(src.shape[0], src.shape[1]).to(torch.int64)
		positional_encoding = positional_encoding.to(self.device)
		for i in range(src.shape[0]):
			positional_encoding[i,:] = torch.LongTensor([list(range(0,src.shape[1]))])

		positional_embeddings = self.pos_embed(positional_encoding)

		if src_skip_inputs is not None:
			track_skip_embed = self.skip_embed(src_skip_inputs)
			embeddings = torch.cat((track_id_embed, track_feat_embed), dim=2) + positional_embeddings + track_skip_embed
		else:
			embeddings = torch.cat((track_id_embed, track_feat_embed), dim=2) + positional_embeddings

		return embeddings

	def greedy_decoder(self, model, src, bert_output, max_len, start_tgt_sequence):
		encoder_output = model.encode(src, bert_output)
		#what the start token is
		#print("DECODER DEEBUG")
		tgt_tokens = start_tgt_sequence.reshape(start_tgt_sequence.shape[0], 1)		
		#tgt_probs = torch.FloatTensor((start_tgt_sequence.shape[0], self.max_length, self.vocab_size))
		#print(tgt_tokens.shape)

		#tgt_no_peek_mask = self.gen_nopeek_mask(tgt_tokens.shape[1]).to(self.device)
		#ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
		tgt_prob = torch.zeros((start_tgt_sequence.shape[0], self.max_length, self.vocab_size)).to(self.device)
		#print(tgt_prob.shape)

		for i in range(max_len):

			#print(i)

			#tgt_no_peek_mask = self.gen_nopeek_mask(tgt_tokens.shape[1]).to(self.device)
			output = model.decode(encoder_output, tgt_tokens, bert_output)
			#print("OUTPUT")
			#print(output.shape)
			output = self.fc(output)[-1, :,:]
			#print("FC")
			#print(output.shape)
			tgt_prob[:,i,:] = output
			output = F.log_softmax(output, dim=1)
			#print(output.shape)
			#next_token_prob = output
			next_token_pred = torch.argmax(output, dim = 1).reshape(output.shape[0],1)
			#print(next_token_pred.shape)

			#remember this is the input to the model
			if i != (max_len - 1):
				tgt_tokens = torch.cat([tgt_tokens, next_token_pred], dim=1)
				#print(tgt_tokens.shape)
			#tgt_probs[:,i,:] = next_token_prob
			#print(tgt_probs.shape)
			#print(tgt_tokens)
		#print("FINISHED DECODING")
		return tgt_tokens, tgt_prob


	def gen_nopeek_mask(self, length):
		mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask