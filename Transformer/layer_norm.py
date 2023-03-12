import torch
import torch.nn as nn
class LayerNorm(nn.module):
  def __int__(self, feature, eps=1e-6):
    """
    :param feature: self-attention x input size
    :param eps:
    :return:
    """
    super(LayerNorm, self).__init__()
    """
    建立super對象 第一個參數:type , class 第二個參數:type, object
    第二個參數決定bind 到哪個class上，同時決定使用哪個mro
    流程 首先從self拿到mro,找到第一個參數在mro位置，往後+1，看有沒有__init__函數再bind到self
    """
    self.a_2 = nn.Parameter(torch.ones(feature))
    self.b_2 = nn.Parameter(torch.zeros(feature))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True) #?
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / pow((std + self.eps) , 1/2) + self.b_2

class SublayerConnection(nn.Module):
  def __init__(self, size, dropout=0.1): #size?
    super(SublayerConnection, self).__init__()
    self.layer_norm = LayerNorm(size)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x, sublayer):
    """

    :param x: self-attention input
    :param sublayer: self-attention layer
    :return:
    """
    return self.dropout(self.layer_norm(x + sublayer(x)))