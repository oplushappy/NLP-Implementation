# model train
import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from gpt_model import *



def epoch_time(start_time, end_time): # express seconds as minutes and seconds
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_step(model,data_loader,optimizer,criterion,clip=1,print_every=None):  #每一个eopch的训练
    model.train() # train mode

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # It is reset every time it is printed, counts the loss within a certain number of batches (default 10), and prints once every 10 batches

    epoch_loss = 0 #epoch的总loss

    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)): #dec_inputs: [b, tgt_len] , dec_outputs: [b, tgt_len]
        optimizer.zero_grad()
        dec_inputs, dec_outputs =dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]       tgt_len<=30

        # with torch.cuda.amp.autocast(): # half-precision training
        outputs, dec_self_attns = model(dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1)) #outputs :(b * tgt_len, vocab_size),dec_outputs.view(-1) :(b * tgt_len)       tgt_len<=300


        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward() # gradient backpropagation


        # Gradient clipping to prevent gradient explosion. If the loss exceeds the clip, reduce the gradient value to one of the original (loss/clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step() # Update model weights

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)

def train(model,data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device) #loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #optimzer

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=100) #train a epoch
        end_time = time.time()

        torch.save(model.state_dict(), r'weights\01\GPT2-%d.pt'%epoch) # save weigh

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  # show as m:s
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')



def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(

        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

if __name__ == '__main__':
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()

    # print(len(datas))  print total data number
    train_data = make_data(datas[::]) #take part of dataset.txt to deal with('\t' replace with <sep>)，return processed list
    # If you want to test whether it can run, you need to set the number of slice steps to -1 first, so that you can extract the longest vector group first, and if the longest vector group does not burst the memory, you can run it.

    train_num_data = [[word2id[word] for word in line] for line in train_data] # replace every word with corresponding id

    batch_size = 22 # By measuring 4g gpu memory can be set to 22, and 6g gpu memory can be set to 32
    epochs = 30
    dataset = MyDataSet(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch) #Call collate_fn for each batch separately, because the sentences in the batch are of different lengths, so the default method of torch cannot be directly used
    # Batch taken out each time, the shape [b, decoder_input_maxlen], [b, decoder_output_maxlen] type is torch.long. decoder_output_maxlen is the longest length in this batch, and the value of each different batch is also different
    # Why not use shuffle=True, because if it is used, each data will be randomly selected in the batch. Some of the random paragraphs are long and some are short, so the short paragraphs will be filled with many <pad>. If the paragraphs in the batch are all about the same length, the efficiency of training can be improved.

    model = GPT().to(device)

    # model.load_state_dict(torch.load('GPT2.pt'))  #get weight

    train(model,data_loader)
