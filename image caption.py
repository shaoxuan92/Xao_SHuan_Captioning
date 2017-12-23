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
    ##########################

    ### DATA PREPROCESSING ###

    ##########################
    load_data,prepare_data=load_dataset(dataset)#we can see load_data,prepare_data as a function to call flick8k.py
    train,valid,test,worddict=load_data()
    #process word_dic
    word_dict = {}
    for i,j in worddict.iteritems():
        word_dict[j]=i
    word_dict[0]='<EOS>'
    word_dict[1]='UNK'
    ##########################

    ### build the theano graph ###

    ##########################
