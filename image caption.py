'''
models incorporated with structure, variety enchancement and new attention model
structure:structured rnn + structured attentional network
variety enhancement: learning method with two terms: one for correctness and one for variety
attention model: refer to paperweekly
'''
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as tensor
from theano.sandbox_rng_mrg import MRG_RandomStreams as RandomStreams
dataset = {'flickr8k':(flickr8k.load_data,flickr8k.prepare_data) }
def load_dataset(name):
    return dataset[name][0],dataset[name][1]

def validate_options(options):
    # get some checks here

    return options
def init_params(options):

    return params


################################

### initialize theano shared ###
### variables according to ###
### the initial parameters ###

###############################
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk]= theano.shared(params[kk],name=kk)#name defines the name of this theano variable
    return tparams
################################

### build a training model ###
### build the entire graph ###
### used for training ###

###############################
def build_model(tparams,options,sampling=True):
    '''
    :param tparams: OrderedDict
        maps names of variables to theano shared variables
    :param options:
        big dictionary with all the settings and hyperparameters
    :param sampling:
        True for stochastic attention(Hard Attention)
    :return:
        trng: theano random generator
            used for dropout, stochastic attention, etc
        use_noise: theano shared variable
            flag that toggles noise on and off
        [x,mask,ctx]: theano variable
            represent the captions,binary mask, and annotations
            for a single batch
        alphas: theano variables
            attention weights
        alpha_sample: theano variable
            Sampled attention weights used in Reinforce for
            stochastic attention
        cost: theano variable
            negative log likelihood
        opt_outs: OrdereDict
            extra outputs required depending on configuration in options
    '''
    trng = RandomStreams(12345)
    use_noise = theano.shared(np.float32(0.))



'''
train
'''
def train(dataset = 'flickr8k',
          max_epochs = 5000,
          batch_size = 16):
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
    # 1. parameters initialization
    models_option = locals.copy()
    models_option = validate_options(models_option)
    params = init_params(models_option)
    # numpy arrays -> theano shared variables for
    tparams = init_tparams(params)
    # build the model for training
    build_model(tparams,models_option)
