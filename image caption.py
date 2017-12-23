'''
models incorporated with structure, variety enchancement and new attention model
structure:structured rnn + structured attentional network
variety enhancement: learning method with two terms: one for correctness and one for variety
attention model: refer to paperweekly
'''
import numpy as np
dataset = {'flickr8k':(flickr8k.load_data,flickr8k.prepare_data) }
def load_dataset(name):
    return dataset[name][0],dataset[name][1]

'''
train
'''
def train(dataset = 'flickr8k'):
    load_data,prepare_data=load_dataset(dataset)#we can see load_data,prepare_data as a function to call flick8k.py