# Credit: Code is Based on Deep Learning Assignment 4 Transfomer Submitted Code (Specifically Ahmad Khan's Submission)

import numpy as np
import torch
from torch import nn
import random

####### Do not modify these imports.

class Transformer(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=20):
        '''
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(Transformer, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################

        self.embedding_layer = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.positional_embedding = nn.Embedding(self.max_length, self.word_embedding_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1

        '''
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        '''

        self.multi_head_attention_layer_1 = MultiHeadAttention(self.num_heads, self.hidden_dim, self.dim_k, self.dim_v, self.dim_q)
        self.multi_head_attention_layer_2 = MultiHeadAttention(self.num_heads, self.hidden_dim, self.dim_k, self.dim_v, self.dim_q)
        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        self.linear_layer_1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final_linear_layer = nn.Linear(self.hidden_dim, self.output_size)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        outputs = None
        x = self.embed(inputs)
        #x = self.multi_head_attention(x)
        x = self.multi_head_attention_layer_1(x)
        x = self.multi_head_attention_layer_2(x)
        x = self.feedforward_layer(x)
        outputs = self.final_layer(x)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################

        input_embeddings = self.embedding_layer(inputs)
        positional_encoding = torch.zeros(inputs.shape[0], inputs.shape[1]).to(torch.int64)
        for i in range(inputs.shape[0]):
            positional_encoding[i,:] = torch.LongTensor([list(range(0,inputs.shape[1]))])

        positional_embeddings = self.positional_embedding(positional_encoding)
        embeddings = input_embeddings + positional_embeddings

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
    
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################

        outputs = None

        q1=self.q1(inputs)
        k1=self.k1(inputs)
        v1=self.v1(inputs)
        k1 = k1.permute(0, 2, 1)

        q2=self.q2(inputs)
        k2=self.k2(inputs)
        v2=self.v2(inputs)
        k2 = k2.permute(0, 2, 1)

        attn_head_1 = torch.matmul(self.softmax(torch.matmul(q1, k1)/torch.sqrt(torch.FloatTensor([[self.dim_k]]))), v1)
        attn_head_2 = torch.matmul(self.softmax(torch.matmul(q2, k2)/torch.sqrt(torch.FloatTensor([[self.dim_k]]))), v2)
        
        attn_concat = torch.cat((attn_head_1, attn_head_2), dim=-1)
        attn_concat = self.attention_head_projection(attn_concat)

        outputs = self.norm_mh(attn_concat + inputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = None
        x = self.linear_layer_1(inputs)
        x = self.relu(x)
        x = self.linear_layer_2(x)
        outputs = self.norm_ff(inputs+x)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################

        outputs = self.final_linear_layer(inputs)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_dim, dim_k, dim_v, dim_q):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

    def forward(self, inputs):
        outputs = None

        q1=self.q1(inputs)
        k1=self.k1(inputs)
        v1=self.v1(inputs)
        k1 = k1.permute(0, 2, 1)

        q2=self.q2(inputs)
        k2=self.k2(inputs)
        v2=self.v2(inputs)
        k2 = k2.permute(0, 2, 1)

        attn_head_1 = torch.matmul(self.softmax(torch.matmul(q1, k1)/torch.sqrt(torch.FloatTensor([[self.dim_k]]))), v1)
        attn_head_2 = torch.matmul(self.softmax(torch.matmul(q2, k2)/torch.sqrt(torch.FloatTensor([[self.dim_k]]))), v2)

        attn_concat = torch.cat((attn_head_1, attn_head_2), dim=-1)
        attn_concat = self.attention_head_projection(attn_concat)

        outputs = self.norm_mh(attn_concat + inputs)

        return outputs   

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True