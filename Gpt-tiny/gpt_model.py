# construct dataset and model
import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import random
import time
from tqdm import tqdm

device = torch.device("cuda:0") #use which gpu
dict_datas = json.load(open('dict_datas.json', 'r')) # read json，include word2id dict & id2word list
word2id, id2word = dict_datas['word2id'], dict_datas['id2word'] 
vocab_size = len(word2id) #how many words in dict 
max_pos = 300  # A paragraph max to 300 word
d_model = 768  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
CLIP = 1 #?

print('Dataset dictionary total: %d word'%vocab_size)


# Input the list of dataset.txt files that have been read, and output a list that replaces tab characters with <sep>
def make_data(datas):
    train_datas = []
    for data in datas:
        data = data.strip() # remove spaces
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>'] # replace tabs with <sep>
        train_datas.append(train_data)

    return train_datas


# define dataset
class MyDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas # This list is to take a part of dataset.txt first, process it (replace tab characters with <sep>), return the processed list, and then replace each word in the list with the corresponding id number.

    def __getitem__(self, item): # Take out a piece of data (a dialogue) from the above list according to the item index, construct the input and output of gpt, pack it into a dictionary and return
        data = self.datas[item]  # Take out a piece of data (a conversation) by item index from the above list
        decoder_input = data[:-1] # Input and output are staggered by one position
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)    # The length of this sentence, in fact, the input and output lengths are the same
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    # This method will be used as the parameter of collate_fn of DataLoader. The guess is because if you don’t write this, torch will call the default collate_fn,
    # which is to convert the data in this batch list into a torch matrix, but here the length of each data in the batch is different,
    #  and it cannot be directly converted into a matrix, and an error will be reported.
    def padding_batch(self, batch): #，receive the batch returned by the getitem method
        decoder_input_lens = [d["decoder_input_len"] for d in batch]    # take out the length of each input data (each paragraph) in the batch
        decoder_output_lens = [d["decoder_output_len"] for d in batch]  # take out the length of each output data (each paragraph) in the batch

        decoder_input_maxlen = max(decoder_input_lens)    #The maximum length of a paragraph inside batch
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch: # Fill "<pad>" for each decoder_input and decoder_output data of the current batch, until the maximum length with the one in the batch
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long) #change type
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        return decoder_inputs, decoder_outputs # shape [b,decoder_input_maxlen], [b,decoder_output_maxlen]  type is torch.long




"""
======================================================================================================================================================================
construct model

"""


# Mask out the characters corresponding to <pad> in the data, so that these pads in the softmax of the similarity matrix of Q and K are all 0, and will not be considered by the subsequent V
def get_attn_pad_mask(seq_q, seq_k): # shape all is [b, tgt_len <300]

    batch_size, len_q = seq_q.size()  # len_q = len_k = tgt_len
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token. It is to mask out the characters corresponding to <pad> in the data, so that the softmax of Q and K will not consider these <pad>
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [b, 1, tgt_len], id is 0(the id of <pad>)pos is True，another pos is False。later will make pos = Ture be masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [b, tgt_len, tgt_len]


# The upper triangular matrix mask, this is because when using the current information to predict the next word, the subsequent information cannot be seen.
def get_attn_subsequence_mask(seq): #seq: [b, tgt_len]

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] #[b, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask  # [b, tgt_len, tgt_len] Upper triangular matrix,dtype=torch.uint8


class ScaledDotProductAttention(nn.Module): 
    # QKV
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask): #Q,K,V is same : [b, n_heads, tgt_len, d_k=64]，attn_mask:[b, n_heads, tgt_len, tgt_len]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # Q and K attn scores : [b, n_heads, tgt_len, tgt_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor in value where mask is True.
        # That is, all elements in the scores matrix corresponding to attn_mask=1 are replaced with -1e9, so that it becomes 0 in the next softmax

        attn = nn.Softmax(dim=-1)(scores) #[b, n_heads, tgt_len, tgt_len]
        context = torch.matmul(attn, V)  # [b, n_heads, tgt_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # d_model=768 ,  d_v = d_k = 64 ,  n_heads=8
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask): #QKV is same : [b, tgt_len, d_model]  , attn_mask: [b, tgt_len, tgt_len]

        residual, batch_size = input_Q, input_Q.size(0)  #
        # [b, tgt_len, d_model] --> [b, tgt_len, d_k * n_heads] -split-> (b, tgt_len, n_heads, d_k) -trans-> (b, n_heads, tgt_len, d_k)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [b, n_heads, tgt_len, d_k=64]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [b, n_heads, tgt_len, d_k=64]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [b, n_heads, tgt_len, d_v=64]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # add n_heads dim and copy tgt_len。attn_mask : [b, n_heads, tgt_len, tgt_len]

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context shape [b, n_heads, tgt_len, d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [b, tgt_len, n_heads * d_v]
        output = self.fc(context)  # [batch_size, tgt_len, d_model]
        return self.layernorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):   # [b,tgt_len,d_model] -> [b,tgt_len,d_model]     输入和输出形状不变
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)  # [batch_size, seq_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        # self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask): #dec_inputs: [b, tgt_len, d_model]    dec_self_attn_mask: [b, tgt_len, tgt_len]

        #dec_outputs: [b, tgt_len, d_model], dec_self_attn: [b, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)  # [b, tgt_len, d_model]
        return dec_outputs, dec_self_attn  # [b, tgt_len, d_model] , [b, n_heads, tgt_len, tgt_len]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # emb matrix shape (vocab_size, 768)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  # Extracting a row in matrix form is more efficient than using mlp directly. Because mlp will have a lot of useless operations
        # emb matrix shape (300,768)   , max_pos can be adjusted
        self.pos_emb = nn.Embedding(max_pos, d_model)     # learnable positional embed    
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs): #input : dec_inputs shape [b,tgt_len]

        seq_len = dec_inputs.size(1) #tgt_len ，the max length of batch，not exceed 300
        pos = torch.arange(seq_len, dtype=torch.long, device=device) #prepare for pos embed，[0,1,2,3,...,seq_len-1]
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [tgt_len] -> [b, tgt_len]

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  # [b, tgt_len, d_model=768]
        # At this time, dec_outputs contains the information of this paragraph and the information of pos.

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [b, tgt_len, tgt_len]  make <pad> be masked
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [b, tgt_len, tgt_len] Upper Tri Matrix
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [b, tgt_len, tgt_len] matrix > 0 all is to be 1 , or be 0

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [b, tgt_len, d_model], dec_self_attn: [b, n_heads, tgt_len, tgt_len], dec_enc_attn: [b, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocab_size) #768->vocab_size,That is, the last hidden layer node 768 is projected onto the nodes of the number of dictionaries

    def forward(self, dec_inputs): #input: dec_inputs shape [b,tgt_len]         tgt_len<=300 (tgt_len is batch max len)
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)  # dec_outpus: [b, tgt_len, d_model=768], dec_self_attns: [n_layers, b, n_heads, tgt_len, tgt_len]
        dec_logits = self.projection(dec_outputs)  # dec_logits: [b, tgt_len, vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns    #left shape [b *tgt_len,vocab_size]

    def greedy_decoder(self, dec_input): #dec_input :[1,tgt_len]   at this time , tgt_len is length of sentence

        terminal = False
        start_dec_len = len(dec_input[0])
        # Keep predicting the next word until the end of "<sep>", if it can not find "<sep>", exit the loop according to the length, and add the "<sep>" at the end
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # forward
            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs) #[1, tgt_len, vocab_size]

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] #[1] is the index, we just need the index. [0] is the probability value, no need      shape: [tgt_len]
            next_word = prob.data[-1] #final word corresponding id
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]: #inspect "<sep>" end
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input   # [1,tgt_len+n]  Because there are n more predicted words

    # random
    def random_decoder(self, dec_input,top_n): #dec_input :[1,tgt_len]   tgt_len length of sentence
        terminal = False
        start_dec_len = len(dec_input[0])
        # Keep predicting the next word until the end of "<sep>", if it can not find "<sep>", exit the loop according to the length, and add the "<sep>" at the end
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # forward
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs) #[1, tgt_len, vocab_size]

            # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] #[1] is the index, we just need the index. [0] is the probability value, no need      shape: [tgt_len]
            # next_word = prob.data[-1] 

            a = projected.to('cpu') #[1, tgt_len, vocab_size]
            b = a.squeeze(0)[-1]  #  [vocab_size]
            c, idx1 = torch.sort(b, descending=True)  # c is the predicted probability, idx1 is the corresponding index
            c = np.array(c[:top_n])**2  # Take the top n most probable
            idx1 = np.array(idx1[:top_n])


            sum = 0 #total prob
            for i in c:
                sum += i # The probability value of top n words

            d = sum * random.uniform(0, 1) # random number

            for i, j in enumerate(c):
                d -= j  # random number - probability value
                if d <= 0:
                    next_word = idx1[i] # the current predict word -> id
                    break

            next_symbol = next_word
            if next_symbol == word2id["<sep>"]: #detect "<sep>" end
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input   # [1,tgt_len+n]  more n predict  word

    def answer(self, sentence): #sentence is we input string [n]   n number of word in sentence
        # Replace the original sentence with the corresponding id, replace \t with the id of "<sep>", 
        # get(word, 1) is to find the corresponding value of word, if the corresponding key is not found, it will return 1 by default, 1 corresponds to <ukn> unknown character
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]  #句子对应的每个字的id号
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0) #[n] -> [1,n]  change type, put to cuda

        # output = self.greedy_decoder(dec_input).squeeze(0)  # [1,n] -> [1,n+m]   # m is the number of newly predicted words, n is the original input question

        output = self.random_decoder(dec_input,top_n=3).squeeze(0)  # [1,n] -> [1,n+m]  m is the number of newly predicted words, n is the original input question

        out = [id2word[int(id)] for id in output]  # Convert the id list to the corresponding word list
        # Count the index of the "<sep>"  in the result
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        # Take the content in the middle of the last two sep as the answer. 
        # The previous one is the input question, which can be discarded directly and does not need to be displayed
        answer = out[sep_indexs[-2] + 1:-1]

        answer = "".join(answer)
        return answer


