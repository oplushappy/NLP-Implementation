
if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = GPT().to(device)
    model.load_state_dict(torch.load(r'weights\01/GPT2-72.pt'))



    model.eval() # detect mode
    #inital input is ''，each time add the information of conversation
    sentence = ''
    while True:
        sentence = ''
        temp_sentence = input("我:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 200:
            # avoid too long
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("GPT-tiny:", model.answer(sentence))