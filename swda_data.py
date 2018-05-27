import pandas as pd 
import numpy as np 
import os
import codecs
import re 
import sklearn

import  collections 

from sklearn.preprocessing import LabelEncoder


train_file = "swda_clip.csv"
vocab_path = "vocab_dict.txt"

over_sample_file = "over_sampling.csv"

def build_vocab_table(data):
    vocab_dict = dict()
    len_dict = dict()
    vocab_counter = collections.Counter()
    max_len = -1
    for line in data:
        line = line.lower()
        line = re.sub(r"['-,.;!:?%$#*&+=\")(_~/]", '', line)
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.split(' ')
        line = [i for i in line if i != '']
        if len(line) not in len_dict.keys():
            len_dict[len(line)] = 1
        else:
            len_dict[len(line)] += 1
        max_len = max(max_len, len(line))
        vocab_counter.update(line)
    vocab_counter = vocab_counter.most_common(19800)
    vocab_counter.append(('unk', 0))
    for w, c in vocab_counter:
        vocab_dict[w] = len(vocab_dict)
    
    with codecs.open(vocab_path, mode = 'w', encoding = 'utf-8') as f:
        for i, (w, c) in enumerate(vocab_counter):
            if i != len(vocab_counter) - 1:
                f.write(w + '\n')
            else:
                f.write(w)

   


    return vocab_dict

def load_vocab_dict():
    vocab_table = dict()
    with codecs.open(vocab_path, mode = 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            
            line = line.strip()
            vocab_table[line] = len(vocab_table)
    return vocab_table

def tokens_to_id(line, vocab_dict):
    sentence = []
    line = line.lower()
    line = line.strip()
    line = re.sub(r"['-,.;!:?%$#*&+=\")(_~/]", '', line)
    line = line.replace('[', '')
    line = line.replace(']', '')
    line = line.split(' ')
    line = [i for i in line if i != '']
    for token in line:
        id = vocab_dict.get(token, vocab_dict['unk'])
        sentence.append(id)

    return sentence

def load_train_data():
    train = pd.read_csv(over_sample_file, sep = ',')
    
    train_data = []
    Y_train = LabelEncoder().fit_transform(train['tags'])
    data = train['utterances'].values
    
    if not os.path.exists(vocab_path) :
        vocab_dict = build_vocab_table(data)
    else:
        vocab_dict = load_vocab_dict()
    for line in data:
        ss = tokens_to_id(line, vocab_dict)
        train_data.append(ss)
    print(train_data[10])
    return train_data, Y_train



if __name__ == '__main__':
    load_train_data()

