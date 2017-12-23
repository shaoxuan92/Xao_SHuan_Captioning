'''
models incorporated with structure, variety enchancement and new attention model
structure:structured rnn + structured attentional network
variety enhancement: learning method with two terms: one for correctness and one for variety
attention model: refer to paperweekly
'''
import numpy as np
def load_data():

#train
def train(
        dataset = 'flickr8k'
            ):
    load_dataset(dataset)