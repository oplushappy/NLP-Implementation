import torch
import math
import torch.nn.functional as F
import torch.nn as nn


def self_attention(query, key, value, dropout=None, mask=None):
  d_k = query.size(-1) #query is matrix
  scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
  #matmul是 tensor乘法
  if mask is not None:
    mask.cuda()
    scores = scores.masked_fill(mask == 0, -1e9)
  self_attn = F.softmax(scores, dim=-1)
  if dropout is not None:
    self_attn = dropout(self_attn)
  return torch.matmul(self_attn, value), self_attn

class MultiHeadAttention(nn.Module):

  def __init__(self):
    super().__init__()
  def forward(self, head, d_model, query, key, value, dropout=0.1,mask=None):
    """

    :param head: 頭數 ， 默認 8
    :param d_model: 輸入的維度 512
    :param query: Q
    :param key: k
    :param value: v
    :param dropout:
    :param mask:
    :return:
    """
    assert (d_model % head == 0)
    self.d_k = d_model
    self.head = head
    self.d_model = d_model

    self.linear_query = nn.Linear(d_model, d_model)
    self.linear_key = nn.Linear(d_model, d_model)
    self.linear_value = nn.Linear(d_model, d_model)

    self.linear_out = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(p=dropout)
    self.attn = None

    if mask is not None:
      mask = mask.unsqueeze(1)

    n_batch = query.size(0)
    query = self.linear_query(query).view(n_batch,-1,self.head,self.d_k).transport(1,2)
    key = self.linear_key(key).view(n_batch,-1,self.head,self.d_k).transport(1,2)
    value = self.linear_value(value).view(n_batch,-1,self.head,self.d_k).transport(1,2)

    x, self.attn = self_attention(query,key,value,dropout=self.dropout,mask=mask)
    x = x.transport(1,2).contigous().view(n_batch, -1, self.head * self.d_k)

    return self.linear_out(x)