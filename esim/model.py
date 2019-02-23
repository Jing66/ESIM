"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked

import sys
import pdb
import os

# a quick hack to get the import from other directories working
sys.path.append(os.getcwd()+'/glomo')
from util import utils
SOFTMAX_MASK = -1e30 

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 attn_heads,
                 attn_dense_sz,
                 graph=None,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.attn_heads = attn_heads            # attention heads for self attention
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        if graph is not None:
            self.graph = graph
            n_layers = graph._hparams['n_layers']
            self.mixture_wgt = nn.Parameter(data=torch.ones(n_layers*2), requires_grad=True)

            self.linear_cat1 = nn.Linear(embedding_dim*2, hidden_size, bias=False)
            self.linear_cat2 = nn.Linear(embedding_dim*2, hidden_size, bias=False)
            self._encoding = Seq2SeqEncoder(nn.LSTM,
                                            self.embedding_dim*2,
                                            self.hidden_size,
                                            bidirectional=True)


        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._self_attn = nn.Sequential(nn.Linear(self.hidden_size*2, attn_dense_sz),
                                        nn.Tanh(),
                                        nn.Linear(attn_dense_sz, attn_heads))

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def fuse(self, H, M):
        '''
        H - [b, T, ndim]
        M - [b,T,T]
        return: H_ =  W1[H;HM] * sigmoid(W2[H;HM])
        '''
        weighted_input = torch.transpose(M,1,2)@H           # [b, T, ndim]
        cat_input = torch.cat((H, weighted_input),dim=2)        # [b,T, ndim*2]
        trans1 = self.linear_cat1(cat_input)
        trans2 = self.linear_cat2(cat_input)                    # [b, T, out_dim]
        H_graph = trans1 * torch.sigmoid(trans2)
        return torch.cat((H,H_graph),dim=2) 

    def graph_feature(self, x, mask):
        '''
        Args:
            - x: [b, T, ndim]
            - mask: [b,T,T]. mask[b,t,t]=1 if x[b,t,:] != 0
        Returns: 
            - M: [b,T,T] mixture of affinity matrix G
        '''
        G, G_prod = self.graph.prop_connection(x, mask)
        n_layers = G.shape[1]
        graph_wgt = F.softmax(self.mixture_wgt, dim=0)
        graph_wgt = graph_wgt.view(2,1,n_layers,1,1)
        M = torch.sum(graph_wgt[0]*G.detach(),dim=1) \
                        + torch.sum(graph_wgt[1]*G_prod.detach(),dim=1)
        return M

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)
        
        # add graph transfer
        if hasattr(self, 'graph'):
            premises_mask_graph = utils.get_mask_3d(premises)
            hypotheses_mask_graph = utils.get_mask_3d(hypotheses)
            premises_M = self.graph_feature(embedded_premises, premises_mask_graph)
            hypotheses_M = self.graph_feature(embedded_hypotheses, hypotheses_mask_graph)
            embedded_premises = self.fuse(embedded_premises, premises_M) 
            embedded_hypotheses = self.fuse(embedded_hypotheses, hypotheses_M)
        # pdb.set_trace()

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        

        # self attention
        v_a= self.self_attention(v_ai, premises_mask)
        v_b = self.self_attention(v_bj, hypotheses_mask)
        
        v_a_avg =  torch.sum(v_a,1)/self.attn_heads  
        v_b_avg = torch.sum(v_b,1)/self.attn_heads      

        #v_a_avg = torch.sum(v_a * premises_mask.unsqueeze(1)
        #                                        .transpose(2, 1), dim=1)\
        #    / torch.sum(premises_mask, dim=1, keepdim=True)
        #v_b_avg = torch.sum(v_b * hypotheses_mask.unsqueeze(1)
        #                                          .transpose(2, 1), dim=1)\
        #    / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities

    def self_attention(self, x, mask):
        x_unnormed = self._self_attn(x)
        attn_heads = x_unnormed.shape[-1]
        mask = torch.unsqueeze(mask,-1).expand(x.shape[0], x.shape[1], attn_heads) 
        v = x_unnormed.masked_fill_(mask==0, SOFTMAX_MASK)
        attention = F.softmax(v, dim=1)
        return torch.bmm(torch.transpose(attention,1,2),x)

def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None: 
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
