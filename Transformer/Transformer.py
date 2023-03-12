from copy import deepcopy

import  torch.nn as nn
from .self_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding, WordEmbedding
from .feedforward import PositionWiseFeedForward
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .generator import WordGenerator
class Transformer(nn.Module):
  def __init__(self, vocab, d_feat, d_model, d_ff, n_head, n_layer, dropout, device='cuda'):
    """

    :param vocab:the length of dictionary
    :param d_feat: the dimension of word vector
    :param d_model: the length of word vector
    :param d_ff:
    :param n_head:
    :param n_layer: the layer of encoder and decoder
    :param dropout:
    :param device:
    """
    super().__init__()
    self.vocab = vocab
    self.device = device
    attn = MultiHeadAttention(n_head, d_model, dropout) #沒Q,K,V
    feedforward = PositionWiseFeedForward(d_model, d_ff)
    self.trg_embed = WordEmbedding(vocab, d_model)
    self.pos_embed = PositionalEncoding(d_model, dropout)
    self.encoder = Encoder(n_layer,EncoderLayer(d_model, deepcopy(attn), deepcopy(feedforward), dropout))
    self.decoder = Decoder(n_layer, DecoderLayer(d_model, deepcopy(attn), deepcopy(feedforward), sublayer_num=4, dropout=dropout))
    self.generator = WordGenerator(d_model, vocab)
  def encode(self, src, src_mask):
    x = self.trg_embed(src[0])
    x = self.pos_embed(x)
    x = self.encoder(x,src_mask)
    return x
  def decode(self, trg, memory, src_mask, trg_mask):
    x = self.trg_embed(trg)
    x = self.pos_embed(x)
    x = self.decoder(x, memory, src_mask, trg_mask)
    return x
  def forward(self, src, trg, mask):
    encoding_output = self.encode(src, mask)
    l2r_output = self.l2r_decode(trg, encoding_output)
    l2r_pred = self.word_prob_generator(l2r_output)
    return l2r_pred
