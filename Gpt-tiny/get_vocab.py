
import json

def get_dict(datas):
    word_count = {} # 
    for data in datas:
        data = data.strip().replace('\t','')
        for word in data:
            word_count.setdefault(word,0) # if has no key, will add it to dict with value 0
            word_count[word] += 1
    word2id = {"<pad>":0,"<unk>":1,"<sep>":2}
    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())} # .keys() return all keys
    word2id.update(temp)
    id2word = list(word2id.keys())

    # print rhe frequency
    high = sorted(word_count.items(),key=lambda x:x[1], reverse=True)[:10]
    # use word frequency(value) to sort in descending order
    low = sorted(word_count.items(),key=lambda x:x[1], reverse=False)[:10]
    print('most high frequency 10 word:',high)
    print('most low frequency 10 word:',low)
    
    return word2id,id2word # return new dictionary(word2id) and list(id2word)

if __name__ == '__main__':
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()
    word2id, id2word = get_dict(datas)

    dict_datas = {"word2id":word2id,"id2word":id2word}
    json.dump(dict_datas, open('dict_datas.json','w',encoding='utf-8'))