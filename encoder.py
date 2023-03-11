from .layer_norm import SublayerConnection
import torch.nn as nn
from copy import deepcopy

def clone_module_to_modulelist(module,module_num):
  return nn.ModuleList([deepcopy(module) for _ in range(module_num)])



class EncoderLayer(nn.Module):
  """
  compose of one encoder layer
  MultiHeadAttention -> Add & Norm -> Feed Forward -> Add & Norm
  """
  def __init__(self, size, attn, feed_forward, dropout=0.1):
    """

    :param size: d_model
    :param attn: has been initial MultiHead Attention layer
    :param feed_forward: has been initial Feed Forward layer
    :param dropout:
    """
    super().__init__()
    self.attn = attn
    self.feed_forward = feed_forward

    self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(size, dropout), 2)

  def forward(self, x, mask):
    """

    :param x:
    :param mask:
    :return:
    """
    first_x = self.sublayer_connection_list[0](x, lambda x_attn: self.attn(x,x,x,mask))
    return self.sublayer_connection_list[1](first_x, self.feed_forward)

class Encoder(nn.module):
  """
  construct n layer
  """
  def __init__(self, n, encoder_layer):
    """

    :param n:
    :param encoder_layer:
    """
    super().__init__()
    self.encoder_layer_list = clone_module_to_modulelist(encoder_layer, n)

  def forward(self, x, src_mask):
    """

    :param x:
    :param src_mask: the sign of mask
    :return:
    """
    for encoder_layer in self.encoder_layer_list:
      x = encoder_layer(x, src_mask)
    return x