import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, dim, dropout, max_len=1000):
    super().__init__()

    if dim % 2 != 0:
      raise ValueError("cannot use sin/cos")

    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len).unqueeze(1) #第二維 增加一個維度
    div_term = torch.exp((torch.arange(0, dim, 2,dtype=torch.float) *
                          -(math.log(10000.0)/dim)))

    pe[:,0::2] = torch.sin(position.float() * div_term)
    pe[:,1::2] = torch.cos(position.float() * div_term)
    ###?
    pe = pe.unsqueeze(1)
    self.register_buffer('pe', pe)
    self.drop_out = nn.Dropout(p=dropout)
    self.dim = dim

  def forward(self, emb, step=None):
    emb = emb * math.sqrt(self.dim)

    if step is None:
      emb = emb + self.pe[:emb.size(0)]
    else:
      emb = emb + self.pe[step]
    emb = self.drop_out(emb)
    return emb