with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

train_datas = [] # total list(contain all grapagh) , data which is be deal with will be added
temp_data = '' # a temp string
for line in lines:
    if line != '\n': # it will will be a normal sentence , if not only has '\n' 
        line = line.strip() # Return a copy of the string with leading and trailing whitespace removed.
        temp_data += (line + '\t') # concat line with Tab at end
        # a paragraph will have many tab
        # like every line , use tab to seperate
    else:
        train_datas.append(temp_data) # add string which is be concat to list
        temp_data = ''

# sort by the length of string
train_datas = sorted(train_datas, key=lambda x:len(x))
new_train_datas = [] # new list
for train_data in train_datas:
    if len(train_data) < 300:
        new_train_datas.append(train_data) # origin have '\t' Tab

# new_train_datas = new_train_datas[::2] # if gpu is not enough, use this to take only half data

with open('dataset.txt', 'w', encoding='utf-8') as f:
    for train_data in new_train_datas:
        f.write(train_data+'\n')

print('It is ok, total lines: %d'%len(new_train_datas))
