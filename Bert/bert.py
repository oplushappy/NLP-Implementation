import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Dataset
text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
"""
print(sentences)
['hello how are you i am romeo', 'hello romeo my name is juliet nice to meet you', 'nice meet you too how are you today', 'great my baseball team won the competition', 'oh congratulations juliet', 'thank you romeo', 'where are you going today', 'i am going shopping what about you', 'i am going to visit my grandmother she is not very well']

print(word_list)
['am', 'today', 'grandmother', 'to', 'too', 'thank', 'name', 'shopping', 'romeo', 'going', 'won', 'meet', 'team', 'very', 'is', 'visit', 'well', 'about', 'are', 'oh', 'hello', 'congratulations', 'where', 'what', 'you', 'my', 'baseball', 'competition', 'she', 'the', 'how', 'great', 'not', 'i', 'nice', 'juliet']
print('---------------------------------------')
"""
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split())) # ['hello', 'how', 'are', 'you',...] no repeat
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3} # when sentence is too short , it will be put in end
#make every word has a relative idx
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
#make a word dictionary , enumerate will return idx , and element
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
# sentence has no ,!?
# setences will return a list
# token_list will be a [[idx, idx, idx...],[],[],...]
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)


"""
print(token_list)

[[12, 7, 22, 5, 39, 21, 15],
 [12, 15, 13, 35, 10, 27, 34, 14, 19, 5],
 [34, 19, 5, 17, 7, 22, 5, 8],
 [33, 13, 37, 32, 28, 11, 16],
 [30, 23, 27],
 [6, 5, 15],
 [36, 22, 5, 31, 8],
 [39, 21, 31, 18, 9, 20, 5],
 [39, 21, 31, 14, 29, 13, 4, 25, 10, 26, 38, 24]]

"""

# BERT Parameters
maxlen = 30  # each setence in batch is composed of 30 token
batch_size = 6 # data num = batch size x Iteration
max_pred = 5 # max tokens of prediction
n_layers = 6 # number of encoder layer
n_heads = 12 # MultiHeadAttention
# Hidden State Size就是我們所說的詞嵌入的維度，它是由模型在預訓練階段即規定好的。Hidden State Size越大代表抽取的特徵越多，運算耗費時間更久，但也更有可能獲得更好的效果。BERT的Base版模型的Hidden State Size為768，Large版本則為1024。
d_model = 768 # the dimension of token embedding 、 Segment Embeddings、Position Embedding
d_ff = 768*4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V ; Q dot K = attn x V
n_segments = 2 # Decoder input is composed by how many sentences

# sample IsNext and NotNext to be same in small batch size



# data pre deal
# 1. we need random make or replace 15% token of each sentence
# 2. we need random concat 2 sentence
def make_data():
  batch = []
  positive = negative = 0 # positive represent number of 2 sentence is neighbor
  # we need make positive:negative = 1:1 in each batch
  while positive != batch_size / 2 or negative != batch_size / 2:
    tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
      len(sentences))  # random take 2 index of sentence (every sentence will have aidx)
    tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index] # according idx to get the sentence
    input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
    segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1) # use 0 to behave first sentence, use 1 to behave second sentence
    """
    print(input_ids)
    #[1, 35, 8, 12, 2, 26, 39, 21, 16, 11, 19, 21, 20, 2]
    print("-------------------------------")
    """
    # MASK LM
    # how many word is we need to predict , how many token is mask
    # we set max_pred to make all is relatively centralized
    n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15))) #number
    # candidate masked position, it is each word(token) in the 2 sentence(they are concate)
    cand_maked_pos = [i for i, token in enumerate(input_ids)
                      if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
    shuffle(cand_maked_pos) # make word in sentence be random order
    """
    print(cand_maked_pos)
    [9, 2, 8, 10, 1, 11, 12, 3, 5, 6, 7]
    print('----------------------------')
    """
    masked_tokens, masked_pos = [], []
    # pos is ids(number), cand_maked_pos[0],[1],[2]...[n_pred]
    # it will like masked_pos = [6,5,17,0,0]
    # masked_tokens=[14,9,16,0,0]
    for pos in cand_maked_pos[:n_pred]:
      masked_pos.append(pos)
      masked_tokens.append(input_ids[pos]) # the id in original order
      if random() < 0.8:  # 80%
        input_ids[pos] = word2idx['[MASK]']  # make mask
      elif random() > 0.9:  # 10%
        index = randint(0, vocab_size - 1)  # random index in vocabulary
        while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
          index = randint(0, vocab_size - 1)
        input_ids[pos] = index  # replace
    """
    print([masked_pos])
    # [[9, 2]]
    print('---------------------------------------')
    print([masked_tokens])
    # [[11, 8]]
    print('---------------------------------------')
    """
    # Zero Paddings
    # make each sentence in batch has the same length
    n_pad = maxlen - len(input_ids)
    input_ids.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)

    # Zero Padding (100% - 15%) tokens
    # make mask list in each sentence of batch has the same number
    if max_pred > n_pred:
      n_pad = max_pred - n_pred
      masked_tokens.extend([0] * n_pad)
      masked_pos.extend([0] * n_pad)
    # batch 4 list : 句子
    if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
      batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
      positive += 1
    elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
      batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
      negative += 1
    print(batch)
    """
    [input_ids :[1, 35, 3, 12, 2, 26, 39, 21, 16, 7, 19, 21, 20, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    segment_ids:[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    masked_tokens:　[11, 8, 0, 0, 0], 
    masked_pos：[9, 2, 0, 0, 0], 
    False]]    
    print("------------------")
    """

  return batch


# Proprecessing Finished

batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
  torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
  def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.masked_tokens = masked_tokens
    self.masked_pos = masked_pos
    self.isNext = isNext

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)

#-----------------------------------Model------------------------------
def get_attn_pad_mask(seq_q, seq_k):
  batch_size, seq_len = seq_q.size()
  # eq(zero) is PAD token
  pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
  return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
  """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
  def __init__(self):
    super(Embedding, self).__init__()
    self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding : (num_embeddings(the word number of sentence) , embedding_dim)
    self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding : (the max token in a sentence , embedding_dim)
    self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
    self.norm = nn.LayerNorm(d_model)

  def forward(self, x, seg): # x = tensor 
    seq_len = x.size(1)
    pos = torch.arange(seq_len, dtype=torch.long) # tensor([0,1,2,...,seq_len-1])
    pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, [seq_len]]
    embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
    return self.norm(embedding) # why


class ScaledDotProductAttention(nn.Module):
  # matmul -> scale -> mask -> softmax -> matmul
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
    scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
    attn = nn.Softmax(dim=-1)(scores) # also can set dim = 2, make a softmax of each dimension row
    context = torch.matmul(attn, V)
    return context


class MultiHeadAttention(nn.Module):
  def __init__(self):
    super(MultiHeadAttention, self).__init__()
    self.W_Q = nn.Linear(d_model, d_k * n_heads) # in_feature , out_feature
    self.W_K = nn.Linear(d_model, d_k * n_heads)
    self.W_V = nn.Linear(d_model, d_v * n_heads)

  def forward(self, Q, K, V, attn_mask):
    # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
    residual, batch_size = Q, Q.size(0)
    # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
    k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
    v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

    attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

    # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
    context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
    # make a right format
    context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                        n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
    output = nn.Linear(n_heads * d_v, d_model)(context) # after concat
    return nn.LayerNorm(d_model)(output + residual)  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
  def __init__(self):
    super(PoswiseFeedForwardNet, self).__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
    return self.fc2(gelu(self.fc1(x))) # in original, w2(relu(w1(layer_norm(x))+b1))+b2 , here x has been Layer_Norm


class EncoderLayer(nn.Module):
  # MultiHead Attention -> Add & Norm -> Feed Forward -> Add & Norm
  def __init__(self):
    super(EncoderLayer, self).__init__()
    self.enc_self_attn = MultiHeadAttention()
    self.pos_ffn = PoswiseFeedForwardNet()

  def forward(self, enc_inputs, enc_self_attn_mask):
    enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V, original parameter is(Q,K,V,attn_mask)
    enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
    return enc_outputs


class BERT(nn.Module):
  def __init__(self):
    super(BERT, self).__init__()
    self.embedding = Embedding() #word + pos embed
    self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    # extra for endoer of Transformer
    self.fc = nn.Sequential( 
      nn.Linear(d_model, d_model),
      nn.Dropout(0.5),
      nn.Tanh(),
    )
    self.classifier = nn.Linear(d_model, 2)
    self.linear = nn.Linear(d_model, d_model)
    self.activ2 = gelu
    # fc2 is shared with embedding layer
    embed_weight = self.embedding.tok_embed.weight
    self.fc2 = nn.Linear(d_model, vocab_size, bias=False) # vocab_size is the total word number in a sentence
    self.fc2.weight = embed_weight

  def forward(self, input_ids, segment_ids, masked_pos):
    output = self.embedding(input_ids, segment_ids)  # [bach_size, seq_len, d_model]
    enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]
    for layer in self.layers:
      # output: [batch_size, max_len, d_model]
      output = layer(output, enc_self_attn_mask)
    # it will be decided by first token(CLS)
    h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
    logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

    masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
    h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
    h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
    logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
    return logits_lm, logits_clsf


model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

# Train
for epoch in range(180):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)