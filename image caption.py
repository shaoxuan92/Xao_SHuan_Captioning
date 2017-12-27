#_*_coding:utf-8_*_
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
def _p(pp, name):
    return '%s_%s' % (pp, name)
def ortho_weight(ndim):
    W = np.random.randn(ndim,ndim)
    u,_,_=np.linalg.svd(W)
    return u.astype('float32')
def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nin == nout and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin,nout)
    return W.astype('float32')
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

''''
Neural network layer definitions.
The life-cycle of each of these layers is as follows
    1) The param_init of the layer is called, which creates
    the weights of the network.
    2) The fprop is called which builds that part of the Theano graph
    using the weights created in step 1). This automatically links
    these variables to the graph.
Each prefix is used like a key and should be unique
to avoid naming conflicts when building the graph.

'''
layers = {'ff':('param_init_fflayer','fflayer'),
          'lstm':('param_init_lstm','lstm_layer'),
          'lstm_cond':('param_init_lstm_cond','lstm_layer_cond')}
def get_layer(name):
    fns = layers[name]
    return(eval(fns[0]),eval(fns[1]))

# feedforward layer: affine transformation + point-wise nonlinearity
# make prefix-appended name


def param_init_fflayer(options,params,prefix='ff',nin=None,nout=None):
    if nin is None:
        nin = options['dim_proj'] ######where is dim_proj
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin,nout,scale=0.01)
    params[_p(prefix,'b')] = np.zeros((nout,)).astype('float32')

def fflayer(tparams,state_below,options,prefix='ronv',activ='lambda x:tensor.tanh(x)',**kwargs):#########state_below?????????
    return eval(activ)(tensor.dot(state_below,tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])
#conditional LSTM layer with Attention
def param_init_lstm_cond(options,params,pre_fix='lstm_cond',nin=None,dim=None,dimctx=None):
    if nin is None:###what is nin,dim,dimctx mean???????????????????????????????????? THINK IT AS Weight
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
# input to LSTM, similar to the above, we stack the matricies for compactness, do one
# dot product, and use the slice function below to get the activations for each "gate"
    W = np.concatenate([norm_weight(nin,dim)],
                       [norm_weight(nin, dim)],
                       [norm_weight(nin, dim)],
                       [norm_weight(nin, dim)],axis=1)
    params[_p(pre_fix,"W")] = W


# LSTM to LSTM（可以理解成lstm是多层的）
    U = np.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(pre_fix,'U')] = U

# bias to LSTM
    params[_p(pre_fix,'b')] = np.zeros((4*dim,)).atype('float32')

# context to LSTM
    Wc = norm_weight(dimctx,dim*4)
    params[_p(pre_fix,'Wc')] = Wc

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

#description: two dimensional??????
    x = tensor.matrix('x',dtype='int64')
    mask = tensor.matrix('mask',dtype='float32')
#context: three dimensional??????
    ctx = tensor.tensor3('ctx',dtype='float32')

    n_timestep = x.shape[0]
    n_samples = x.shape[1]

# encoder
    if options['lstm_encoder']:
        pass
    else:
        ctx0 = ctx

# initial state/cell
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1,options['n_layers_init']): # pay attention it starts at '1'
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_init_%d' % lidx, activ='rectifier')

# [1] denotes that it only call the second function:fflayer
    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
#lstm decoder
    attn_updates = []
    proj, updates = get_layer('lstm_cond')[1](tparams,
                                       emb,
                                       options,
                                       prefix='decoder',
                                       mask=mask, context=ctx0,
                                       one_step=False,
                                       init_state=init_state,
                                       init_memory=init_memory,
                                       trng=trng,
                                       use_noise=use_noise,
                                       sampling=sampling))




'''
train
'''
def train(dataset = 'flickr8k',
          max_epochs = 5000,
          batch_size = 16,
          lstm_encoder=False, # if true,run biLSTM on input units
          n_layers_init = 1,
          dim = 1000
        ):
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
