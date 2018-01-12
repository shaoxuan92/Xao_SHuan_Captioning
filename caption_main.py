from collections import OrderedDict
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor


# some useful shorthands
def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])


# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    # !!INPUT!! to LSTM
    W = np.concatenate([norm_weight(nin,dim), # nin = word_dim = 100, dim = dim = 1000
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W # (100, 4000)

    # !!LSTM!! to LSTM
    U = np.concatenate([ortho_weight(dim), # this is also(dim, dim) dim = dim = 1000
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U # (1000, 4000)

    # bias to LSTM
    params[_p(prefix,'b')] = np.zeros((4 * dim,)).astype('float32')

    # !!CONTEXT!! to LSTM
    Wc = norm_weight(dimctx,dim*4) # nin = dimctx = 512, dim = dim * 4 = 4000
    params[_p(prefix,'Wc')] = Wc # (512, 4000)

    # ATTENTION WEIGHT
    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, ortho=False) # nin = dimctx = 512
    params[_p(prefix,'Wc_att')] = Wc_att # (512, 512)

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx) # nin = dim = 1000, nout = dimctx = 512
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = np.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx,1) # nin = nout = dimctx = 512
    params[_p(prefix,'U_att')] = U_att
    c_att = np.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def lstm_cond_layer(tparams,state_below,options,prefix,mask,context,one_step,init_memory,init_state,trng,use_noise,sampling,argmax=False,**kwags):
    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] #########################################################################

    # infer lstm dimension, it seems that both are correct.
    dim = tparams[_p(prefix,'U')].shape[0]
    dim = options['dim']

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')] # context:batch_size * 16 regions * 512 dimension of region representation

    # projected input
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix,'b')]

    # additional parameters for stochastic soft attention
    if options['attn_type'] == 'stochastic':
        temperature = options.get('temperature', 1)
        semi_sampling_p = options.get('semi_sampling_p', 1)
        temperature_c = theano.shared(np.float32(temperature),name='temperature_c')
        h_sampling_mask = trng.binomial((1,), p=semi_sampling_p, n=1, dtype=theano.config.floatX).sum()

    #functions in theano.scan
    def _step(m_, x_, h_, c_, a_, as_, ct_, pctx_):
        # prepare for calculating attention distribution
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        # IT IS REALLY ADD OR JUST CONCATENATE
        pctx_ = pctx_ + pstate_[:,None,:]# the initial of pctx_is None, but I think it is from non_sequnces instead of outputs_info. Pay more attention to line 496-line 509 of where outputs_info come from
        pct_list = []
        pct_list.append(pctx_)
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')]) + tparams[_p(prefix,'b_att')] # batch_size * 16 regions represent the distribution of each region
        alpha_pre = alpha
        alpha_shp = alpha.shape
        alpha = tensor.nnet.softmax(temperature_c*alpha.reshape([alpha_shp[0],alpha_shp[1]]))# I think reshape is useless########################################
        alpha_sample = h_sampling_mask * trng.multinomial(pvals=alpha, dtype=theano.config.floatX) + (1. - h_sampling_mask) * alpha ## some kind of tricky thing

        # caculate the context used for caption generation
        ctx_ = (context * alpha_sample[:,:,None]).sum(1) ##############################what exactly is it........

        # IT IS REALLY ADD OR JUST CONCATENATE
        # I think it is 'concatenation of previous [hidden_state, input_word, context_vector] used for caculate next time activations of LSTM
        preact = tensor.dot(h_,tparams(_p(prefix, 'U')))
        preact += x_
        preact += tensor.dot(ctx_,tparams(_p(prefix,'Wc')))

        #compute next time activations of LSTM

        
    # something used in theano.scan for building recurrent graph
    _step0 = lambda m_, x_, h_, c_, a_, as_, ct_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)
    seqs = [mask, state_below]#################################################
    outputs_info = [init_state,init_memory,tensor.alloc(0., n_samples, pctx_.shape[1]),
                        tensor.alloc(0., n_samples, pctx_.shape[1]),
                        tensor.alloc(0., n_samples, context.shape[2])]#####################################
    rval, update = theano.scan(_step0,
                               sequences=seqs,
                               outputs_info=outputs_info,
                               non_sequences=[pctx_],
                               n_steps=state_below.shape[0])
    return rval, update


    pass

def build_model(tparams, options, sampling=True):
    trng = RandomStreams(1245)
    use_noise = theano.shared(np.float32(0.))

    # captions of images: #words in the caption * #number of images
    x = tensor.matrix('x',dtype='int64')
    # mask
    mask = tensor.matrix('mask',dtype='float32')
    # context: #number of images * #number of annotations(image regions/patch) * #the dimension of representation of each region
    ctx = tensor.tensor3('ctx',dtype='float32')

    # change each word in x into a vector representation of each word
    emb = tparams['Wemb'][x.flatten()].reshape([x.shape[0],x.shape[1],options['word_dim']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:],emb[:-1])
    emb = emb_shifted

    # initial state and initial cell (all set to 1000-D)
    # to generate initial state and initial cell, we need ctx_mean
    ctx_mean = ctx.mean(1) # (#number of images, the dimension of representation) = (#number of batch, 512)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')# ff_state_W: (512,1000)
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    #
    proj, updates = get_layer('lstm_cond')[1](tparams, emb, options, prefix='decoder', mask=mask, context=ctx, one_step=False, init_state=init_state,
                                              init_memory=init_memory, trng=trng, use_noise=use_noise, sampling=sampling)
    pass



def init_params(options):
    params = OrderedDict()
    # word embedding (change word to one specific dimension) Ey
    params['Wemb'] = norm_weight(options['dict_size'], options['word_dim'])

    # context vector dimension (change !!IMAGE!! context to one specific dimension (relative to Z)
    # params['ctx_dim'] = options['ctx_dim']
    ctx_dim = options['ctx_dim']

    # parameters for initial memory and hidden states, and for input units of LSTM
    # for lidx in xrange(1, options['n_layers_init']):
    #    params = get_layer['ff'][0](options, params, prefix='ff_init_%dlidx'%lidx,nin=ctx_dim, nout=ctx_dim)
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=options['ctx_dim'], nout=options['dim']) # nin = ctx_dim = 512, nout = dim = 1000
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=options['ctx_dim'], nout=options['dim'])

    # parameters for decoder
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder',nin=options['word_dim'], dim=options['dim'], dimctx=options['ctx_dim'])

    # parameters for compute word probability
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['word_dim'])########why not directly n_words
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['word_dim'], nout=options['dict_size'])
    return params


def train(max_epoch=4000,
          dict_size=10000,
          word_dim=100,
          ctx_dim=512,
          n_layers_init=1, # number of layers to initialize LSTM on input units
          dim=1000,
          attn_type = 'stochastic',
          temperature = 1,
          semi_sampling_p = 0.5
          ):

    print 'start function train()'
    # load parameters
    model_options = locals().copy()
    print model_options
    # load data and dictionary

    # build parameters for the model
    params = init_params(model_options)  # ORDERED DICTIONARY
    tparams = init_tparams(params)
    trng, use_noise, inps, alphas, alphas_sample, cost, opt_outs = build_model(tparams, model_options)



if __name__ == '__main__':
    train()
