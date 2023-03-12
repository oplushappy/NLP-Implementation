from .layer_norm import SublayerConnection
from .encoder import  clone_module_to_modulelist
import torch.nn as nn
from copy import deepcopy

class DecoderLayer(nn.Module):
  """
  compose of one decoder layer
  Mask MultiHeadAttention -> Add & Norm-> MultiHead Attention-> Add & Norm -> Feed Forward -> Add & Norm
  """
  def __init__(self, d_model, attn, feed_forward, sublayer_num, dropout=0.1):
    """

    :param d_model:
    :param attn:
    :param feed_forward:
    :param sublayer_num:
    :param dropout:
    """
    super().__init__()
    self.attn = attn
    self.feed_forward = feed_forward
    self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(d_model, dropout), 2)

  def forward(self, x, memory, trg_mask):
    """

    :param x:
    :param memory:
    :param mask: encoder output
    :return:
    """

    first_x = self.sublayer_connection_list[0](x, lambda x_attn: self.attn(x,x,x,trg_mask))
    second_x = self.sublayer_connection_list[1](first_x, lambda second_x_attn: self.attn(first_x,memory,memory,None))
    return self.sublayer_connection_list[-1](second_x, self.feed_forward)

class Decoder(nn.module):
  """
  construct n layer
  """
  def __init__(self, n_layers, decoder_layer):
    """

    :param n_layers:
    :param decoder_layer:
    """
    super().__init__()
    self.decoder_layer_list = clone_module_to_modulelist(decoder_layer, n_layers)

  def forward(self, x, memory, trg_mask):
    """

    :param x:
    :param src_mask: the sign of mask
    :return:
    """
    for decoder_layer in self.decoder_layer_list:
      x = decoder_layer(x, memory, trg_mask)
    return x