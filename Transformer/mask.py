import numpy as np
import torch

def src_trg_mask(src, r2l_trg, trg, pad_idx):
  """

  :param src: encoder input
  :param r2l_trg: r2l 方向 decoder input
  :param trg: l2r 方向 decoder input
  :param pad_idx: pad index
  :return: trg not exist , return the mask of encoder input;
  """
  if trg and r2l_trg:
    trg_mask = (trg != pad_idx).unsqueeze(1) & sequence_mask(trg.size(1)).type_as(src_image_mask.data)

def sequence_mask(size):
  """

  :param size: 生成詞個數
  :return: 右上三角為1, 左下三角為0,上三角矩陣
  """
  attn_shape = (1,size,size)
  mask = np.triu(np.ones(attn_shape),k=1).astype('unit8')
  return (torch.from_numpy(mask) == 0).cuda()