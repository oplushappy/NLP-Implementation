import torch.nn as nn
import torch.nn.functional as F
class WordGenerator(nn.Module):
  def __init__(self, d_model, vocab_size):
    """

    :param d_model: 詞向量維度
    :param vocab_size: 詞典大小
    """
    super().__init__()
    self.linear = nn.Linear(d_model,vocab_size)

    def forward(self, x):
      return F.log_softmax(self.linear(x), dim=-1)