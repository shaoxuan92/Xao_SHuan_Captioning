# -*- coding: utf-8 -*-
'''
models incorporated with structure, variety enchancement and new attention model
structure:structured rnn + structured attentional network
variety enhancement: learning method with two terms: one for correctness and one for variety
attention model: refer to paperweekly
'''
import numpy as np
import cPickle as pkl
from collections import OrderedDict
import theano
import theano.tensor as tensor
from Homogenous_data import HomogeneousData
from theano.sandbox_rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.cross_validation import kFold
import time
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

def fflayer(tparams,state_below,options,prefix='ronv',activ='lambda x:tensor.tanh(x)',**kwargs):
#########state_below是一个3D矩阵，[n_step,Batch_size,dim_word]
    return eval(activ)(tensor.dot(state_below,tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

def param_init_lstm(options,params,prefix='lstm',nin=None,dim=None):
    W = np.concatenate([norm_weight(nin,dim),norm_weight(nin,dim),norm_weight(nin,dim),norm_weight(nin,dim)],axis=1)
    params[_p(prefix,'W')] = W
    U = np.concatenate([ortho_weight(dim),ortho_weight(dim),ortho_weight(dim),ortho_weight(dim)],axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = np.zeros((4 * dim,)).astype('float32')
    return params
def lstm_layer(tparams,state_below,options,prefix='lstm',mask=None,**kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]
    if state_below.ndim ==3:
        n_samples = state_below.shape[1]
        init_state = tensor.alloc(0., n_samples, dim)
        init_memory = tensor.alloc(0., n_samples, dim)
    #during sampleing
    else:
        n_samples = 1
        init_state = tensor.alloc(0.,dim)
        init_memory = tensor.alloc(0.,dim)
    if mask == None:
        mask = tensor.alloc(1.,state_below.shape[0],1)

    # use the slice to calculated the diffrent gate
    def _slice(_x,n,dim):
        if _x.ndim == 3:
            return _x[:,:, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:,n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]
    def _step(m_,x_,h_,c_):
        preact = tensor.dot(h_,tparams[_p(prefix,'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact,0,dim))
        f = tensor.nnet.sigmoid(_slice(preact,1,dim))
        o = tensor.nnet.sigmoid(_slice(preact,2,dim))
        c = tensor.tanh(_slice(preact,3,dim))
        c = f * c_ + i * c
        h = o * tensor.tanh(c)
        return h, c, i, f, o, preact
    state_below = tensor.dot(state_below, tparams[_p(prefix,'W')]) + tparams[_p(prefix,'b')]
    rval, updates = theano.scan(_step,sequence=[mask,state_below],outputs_info = [init_state,init_memory,None,None,None,None],name=_p(prefix,'_layers'),
                                n_steps=nsteps,profile=False)
    return rval
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

# ATTENTION
######################################

###### TO BE DONE #####################

#######################################

    return params
def lstm_cond_layer(tparams,state_below,options,prefix='lstm',mask=None,
                    context=None,one_step=False,init_memory=None,
                    init_state=None,trng=None,use_noise=None,sampling=True,
                    argmax=False,**kwargs):
    assert context, 'context must be provided'
    ## 所以说state_below 第一维度是句子的长度？？？第二维度是batch_size？？？
    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    if mask is None:
        mask = tensor.alloc(1.,state_below.shape[0],1)

    # infer lstm dimension
    #注意prefix_W：是input weights,,,prefix_U是previous hidden activation
    dim = tparams[_p(prefix,'U')].shape[0]########????
    #### initial/previous state???????????????????????????
    if init_state is None:##虽然这个函数默认参数是None，但是要看调用时，所以
        init_state = tensor.alloc(0.,n_samples,dim)
    #### initial/previous memory?????????????????????????
    if init_memory is None:
        init_memory = tensor.alloc(0.,n_samples,dim)
    ###projected context，也就是把context弄成某个维度。
    #Wc_attn是context-->hidden。
    pctx_ = tensor.dot(context,tparams[_p(prefix,'Wc_attn')] + tparams[_p(prefix,'b_attn')])
    ####proected x
    ####state_below ??????????????????????????????
    state_below = tensor.dot(state_below,tparams[_p(prefix,'W')]) + tparams[_p(prefix,'b')]
    ###### attentional parameters for stochastic hard attention
    if options['attn_type'] == 'stochastic':
        temperature = options.get("temperature",1)
        semi_sampling_p = options.get("semi_sampling_p",0.5)
        temperature_c = theano.shared(np.float32(temperature),name='temperature_c')
        h_sampling_mask = trng.binomial((1,),p=semi_sampling_p,n=1,dtype=theano.config.floatX).sum()
    # 切片，计算的时候是几个门一起计算，切片将各个门的值分开
    def _slice(_x,n,dim):
        if _x.ndim == 3:
            return _x[:,:,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]
    def _step(m_,x_,h_,c_,a_,as_,ct_,pctx_,dp_=None,dp_attn_=None):
         #attention distribution
         #'Wd_att' attention: LSTM_HIDDEN
        pstate_ = tensor.dot(h_,tparams[_p(prefix,'Wd_att')])
         #pctx_ (pojectedd context))
        pctx_ = pctx_ + pstate_[:,None,:]#相当于numpy.newaix增加一个轴
        pctx_list = [] #pctx_ (pojectedd context))
        pctx_list.append(pctx) #pctx_ (pojectedd context))
        pctx_ = tanh(pctx_) #pctx_ (pojectedd context))
         #u_attn : dimctx
        alpha = tensor.dot(pctx_,tparams[_p(prefix,'U_att')])+tparams[_p(prefix,'c_tt')]
        alpha_pre = alpha
        alpha_shp = alpha.shape
        if options['attn_type'] == 'deterministic':
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]]))
            ctx_ = (context * alpha[:,:,None]).sum(1)###current contxt
            alpha_sample = alpha
        #  x_ = state_below_ = emb*encoder_w+encoder_b
        #preact = h(t-1) * U + x_
        # x_是previous word

    preact = tensor.dot(h_,tparams[_p(prefix,'U')])
    preact += x_
    preact += tensor.dot(ctx_,tparams[_p(prefix,'Wc')])

    i = _slice(preact,0,dim)
    f = _slice(preact,1,dim)
    o = _slice(preact,2,dim)
    i = tensor.nnet.sigmoid(i)
    f = tensor.nnet.sigmoid(f)
    o = tensor.nnet.sigmoid(o)
    c = tensor.tanh(_slice(preact,3,dim))

    ### compute the new memory and hidden states
    c = f * c_ + i * c
    c = m_[:,None] * h + (1. - m_)[:,None] * h_

    h = o * tensor.tanh(c)
    h = m_[:,None] * h + (1.- m_)[:,None] * h_

    rval = [h,c,alpha,alpha_sample,ctx_]

    if options['use_dropout_lstm']:
        pass
    else:
        if options['selector']:
            pass
        else:
            _step0 = lambda m_, x_, h_, c_, a_, as_, ct_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)
    if one_step:
        pass
    else:
        seqs = [mask,state_below]
        outputs_info = [init_state,
                        init_memory,
                        tensor.alloc(0.,n_samples,pctx_.shape[1]),
                        tensor.alloc(0.,n_samples,pctx_.shape[1]),
                        tensor.alloc(0.,n_samples,context.shape[2])]
        outputs_info += [None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None]+[None]##*options['n_layers_att']
        rval,updates = theano.scan(_step0,
                                   sequences = seqs,
                                   outputs_info = outputs_info,
                                   non_sequence=[pctx_],
                                   name=_p(prefix,'_layers'),
                                   n_steps=nsteps,profile=False)
        return rval, updates
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
# x是int类型的二维矩阵。代表不同的样本比如34个image样本，每个样本对应的caption对应的句子中包含的单词用序号表示[3,5,1000,340]
    x = tensor.matrix('x',dtype='int64')
    mask = tensor.matrix('mask',dtype='float32')
#context: three dimensional（（（ 样本的个数，每个样本中标注的数量，和每个标注的维度）））?????????????????????????????????????????????????????????
    ctx = tensor.tensor3('ctx',dtype='float32')

    n_timestep = x.shape[0]#行代表steps，可以理解成句子的长度（统一到了固定长度）。
    n_samples = x.shape[1]#列代表不同的样本
## 进行word_embedding
# tparams['Wemb']是10000*128维度，10000表示词典大小,ID范围是0-9999，每个单词是128维的向量。
# x是int类型的二维矩阵。代表不同的样本比如34个image样本，每个样本对应的caption对应的句子中包含的单词用序号表示[3,5,1000,340]
    emb = tparams['Wemb'][x.flatten()].reshape([n_timestep,n_samples,options['dim_word']])#dim_word也就是我们说的128维
    emb_shifted = tensor.zeros_like (emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:],emb[:-1])#为emb_shifted进行赋值。结果就是emb_shifted[0]全部是0；其中emb_shifted[1:]是房子的第一层
    emb = emb_shifted#第0行跑到了第一行，第一行跑到了第二行，，目的是decoder的时候，第一个timestep喂给隐藏层的输入时0，然后产生第一个单词，
#然后第一个单词作为输入，产生第二个单词。
# encoder
    if options['lstm_encoder']:
        pass
    else:
        ctx0 = ctx##############################################

# initial state/cell
    ctx_mean = ctx0.mean(1)
    ##ctx和ctx_mean均等于encoder最后一个隐藏层h_t????
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
                                       init_state=init_state,######用到了上面的初始化
                                       init_memory=init_memory,
                                       trng=trng,
                                       use_noise=use_noise,
                                       sampling=sampling)

    attn_updates += updates
    proj_h = proj[0]
    ### optional deep attention
    if options['n_layers_lstm'] > 1:
        pass

    alphas = proj[2]
    alpha_sample = proj[3]
    ctxs = proj[4]
    # compute word probabilities
    # compute the output word probability
    # given the LSTM state, the context vector
    # and the previous word:
    logit = get_layer('ff')[1](tparams,proj_h,options,prefix='ff_logit_lstm',activ='linear')
    logit = tanh(logit)
    # compute softmax
    logit = get_layer('ff')[1](tparams,logit,options,prefix='ff_logit',activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape(logit_shp[0]*logit_shp[1],logit_shp[2]))

    #index into the computed probability to give the log likelihood
    x_flat = x.flatten()
    p_flat = probs.flatten()
    cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e08)
    cost = cost.reshape(x.shape[0],x.shape[1])
    masked_cost = cost * mask
    cost = (masked_cost).sum(0)

    # optional outputs
    opt_outs = dict()
    if options['attn_type'] == 'stochastic':
        opt_outs['masked_cost']=masked_cost
        opt_outs['attn_updates']=attn_updates
    trng, use_noise, [x, mask, ctx], alphas, alpha_sample, cost, opt_outs



#generate sample
def gen_sample(tparams, f_init, f_next, ctx0, options,
               trng=None, k=1, maxlen=30, stochastic=False):

    '''
    generate sample with beam search
    Generate captions with beam search.

    This function uses the beam search algorithm to conditionally
    generate candidate captions. Supports beamsearch and stochastic
    sampling.
    Parameters
    ----------
    tparams : OrderedDict()
        dictionary of theano shared variables represented weight
        matricies
    f_init : theano function
        input: annotation, output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        takes the previous word/state/memory + ctx0 and runs one
        step through the lstm
    ctx0 : numpy array
        annotation from convnet, of dimension #annotations x # dimension
        [e.g (196 x 512)]
    options : dict
        dictionary of flags and options
    trng : random number generator
    k : int
        size of beam search
    maxlen : int
        maximum allowed caption size
    stochastic : bool
        if True, sample stochastically
    Returns
    -------
    sample : list of list
        each sublist contains an (encoded) sample from the model
    sample_score : numpy array
        scores of each sample

    '''
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0
    rval = f_init(ctx0)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
# only matters if we use lstm encoder
    rval = f_init(ctx0)# f_init :Input: annotation, Output: initial lstm state and memory (also performs transformation on ctx0 if using lstm_encoder)
    ctx0 = rval[0]
    next_state = []
    next_memory = []

# the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layer_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1,next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1 + options['n_layers_lstm'] + lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    next_w = -1 * np.ones((1,)).astype('int64')
    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w,ctx0]+next_state+next_memory))
        next_p = rval[0]#rval[0]是一个3D矩阵，[n_Step，BatchSize，Emb_Dim][n_Step，BatchSize，Emb_Dim]。
        next_w = rval[1]
        # extract all the states and memories
        next_state = []
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2 + lidx])
            next_memory.append(rval[2 + options['n_layers_lstm'] + lidx])
    if stochastic:
        pass
    else:

def pred_probs(f_log_probs,options,worddict,prepare_data,data,iterator,verbose=False):
    n_samples = len(data[0])
    probs = np.zeros((n_samples,1)).astype('float32')
    n_done = 0
    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[0][t] for t in valid_index],data[1],worddict,maxlen=None,n_words=options['n_words'])
        pred_probs = f_log_probs(x,mask,ctx)
        probs[valid_index] = pred_probs[:None]

        n_done += len(valid_index)
###############################

##### Build_sampler###########

##############################
def build_sampler(tparams,options,use_noise,trng,ssampling=True):
    '''

    :param tparams:
    :param options:
    :param use_noise:
    :param trng:
    :param ssampling:
    :return:
     f_init: theano function
        Input: annotation, Output:initial state and memory
        (also performs transformation on ctx0 if using lstm encoder
     f_next: theano function
        Takes the previous word/state/memory + ctx0 and run next
        step through the lstm(beam search)

    '''
    #context :
    ctx = tensor.matrix('ctx_sampler',dtype = 'float32')
    #initial state/cell
    ctx_mean = ctx.mean(0)
    for lidx in xrange(1,options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams,ctx_mean,options,
                                      prefix='ff_init_%d'%lidx,activ='rectifier')

    init_state = [get_layer('ff')[1](tparams,ctx_mean,options,prefix='ff_state',activ='tanh')]
    init_memory = [get_layer('ff')[1](tparams,ctx_mean,options,prefix='ff_memory',actic='tanh')]

    # print build f_init
    f_init = theano.function([ctx],[ctx]+init_state+init_memory,name='f_init',profile=False)
    ctx = tensor.matrix('ctx_sampler', dtype='float32')
    x = tensor.vector('x_sampler', dtype='int64')
    init_state = [tensor.matrix('init_state', dtype='float32')]
    init_memory = [tensor.matrix('init_memory', dtype='float32')]

    # for the first word , emb should be all zero
    #Tensor.switch(Bool,Ture Operation,False Operation)
    emb =tensor.switch(x[:,None] < 0, tensor.alloc(0.,1,tparams['Wemb'].shape[1]),tparams['Wemb'][x])
    proj = get_layer('lstm_cond')[1](tparams,emb,options,
                                     prefix='decoder',
                                     mask=None,context=ctx,
                                     one_step=True,
                                     init_state=init_state[0],
                                     init_memory=init_memory[0],
                                     trng=trng,
                                     use_noise=use_noise,
                                     sampling=sampling)
    next_state,next_memory,ctxs=[proj[0]],[proj[1]],[proj[4]]
    proj_h = proj[0]

    if options['use_dropout']:
        pass
    else:
        proj_h = proj[0]

    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    logit = tanh(logit)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pval=next_probs).argmax(1)
    # next word probability
    f_next = theano.function([x,ctx]+init_state+init_memory,[next_probs,next_sample]+next_memory+next_state,name='f_next',profile=False)
    return f_init,f_next
'''
train
'''
def train(dataset = 'flickr8k',
          max_epochs = 5000,
          batch_size = 16,
          lstm_encoder=False, # if true,run biLSTM on input units
          n_layers_init = 1,
          dim = 1000,
##dim_word和n_words是word_embedding用的： 一个是每个单词的维度：dim_word,一个是总的数量n_words，也就是把单词映射成字典数量大小。相当于计算每个单词离这个单词的距离。
          dim_word=100, n_words=1000,
          maxlen=100,
          valid_batch_size = 16,
          lrate=0.01,
          dispFreq=100,
          saveFreq=1000,
          saveto='model.npz',  # relative path of saved model file
          sampleFreq=100,  # generate some samples after every sampleFreq updates

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
    trng,use_noise,inps,alphas,alphas_sample,\
    cost,opt_outs = \
    build_model(tparams,models_option)

    print 'build sampler'
    f_init,f_next = build_sampler(tparams,models_option,use_noise,trng)

    # compute loss  without regularizer
    f_log_probs = theano.function(inps, -cost, profile=False,
                                            updates=opt_outs['attn_updates']
                                            if model_options['attn_type']=='stochastic'
                                            else None)
    cost = cost.mean()

    hard_attn_updates = []
    #back prop!
    if models_option['attn_type'] == 'deterministic':
        pass
    else:
        # shared variables for hard attention
        baseline_time = theano.shared(np.float32(0.), name='baseline_time')
        opt_outs['baseline_time'] = baseline_time
        alpha_entropy_c = theano.shared(np.float32(alpha_entropy_c), name='alpha_entropy_c')
        alpha_entropy_reg = alpha_entropy_c * (alphas * tensor.log(alphas)).mean()
        # [see Section 4.1: Stochastic "Hard" Attention for derivation of this learning rule]
        if models_option['RL_sumCost']:
            grads = tensor.grad(cost, wrt=itemlist(tparams),
                                disconnected_inputs='raise',
                                known_grads={
                                    alphas: (baseline_time - opt_outs['masked_cost'].mean(0))[None, :, None] / 10. *
                                            (-alphas_sample / alphas) + alpha_entropy_c * (tensor.log(alphas) + 1)})
        else:
            pass

            # [equation on bottom left of page 5]
            hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
            # updates from scan
            hard_attn_updates += opt_outs['attn_updates']
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, hard_attn_updates)

    print 'Optimization'

    train_iter = HomogeneousData(train,batch_size=batch_size,maxlen=maxlen)

    if valid:
        kf_valid = Kfold(len(valid[0]),n_folds=len(valid[0])/valid_batch_size,shuffle=False)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    history_errs = []
    best_p = None
    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        print 'Epoch: ',eidx
        n_samples = 0

        for caps in train_iter:
            n_samples += len(caps)
            uidx += 1
            #  turn on dropout
            use_noise.set_value(1.)####通过查询可以知道use_noise是个占位符shared variable

            #processing the caption and record the time to help detect bottleneck
            pd_start = time.time()
            #prepare_data 见 flickr8k.py中的prepare
            x,mask,ctx = prepare_data(caps,#是train_iter的东西，在homogeneousData中获得
                                      train[1],##由flickr8k.py 中load_data
                                      worddict,##同上，也在load_data中
                                      maxlen=maxlen,
                                      n_words=n_words)
            pd_duration = time.time() - pd_start

            # get the loss for the minitbatch and update the weights
            ud_start = time.time()
            cost = f_grad_shared(x,mask,ctx)
            f_update(lrate)
            ud_duration = time.time() - ud_start  # some monitoring for each mini-batch
            ##检查有没有NAN
            if np.isnan(cost) or np.isinf(cost):
                print 'Nan Detected'
                return 1.,1.,1.
            if np.mod(uidx,dispFreq) == 0:
                print 'Epoch', eidx, 'Update', uidx, 'Cost', cost, 'PD', pd_duration, 'UD', ud_duration
            # Checkpoint
            if np.mod(uidx,saveFreq) == 0:
                print 'Saving...'
                if best_p is not None:
                    pass
                else:
                    params = unzip(tparams)
                np.savez(saveto,history_errs=history_errs,**params)#。savez()提供了将多个数组存储至一个文件的能力
                pkl.dump(models_option,open('%s.pkl'%saveto,'wb'))

                print 'Done...'

            # Print a generated sample as a sanity check
            if np.mod(uidx,sampleFreq) == 0:
                #首先turn out drop-out
                use_noise.set_value(0.)
                x_s  = x
                mask_s = mask
                ctx_s = ctx

                for jj in xrange(np.minimum(10,len(caps))):
                    sample,score = gen_sample(tparams,f_init,f_next,ctx_s[jj],models_option,trng=trng,k=5,maxlen=30,stochastic=False)
                    # Decode the sample from encoding back to words
                    print 'Truth ',jj,': '
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_dict:
                            print word_dict[vv]
                        else:
                            print 'UNK'
                    print
                    for kk,ss in enumerate([sample[0]]):
                        print 'Sample (',kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_dict:
                                print worddict[vv],
                            else:
                                print 'UNK'
                    print

            if np.mod(udix, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if valid:
                    valid_err = -pred_probs(f_log_probs,models_option,worddict,prepare_data,valid,kf_valid).mean()
                if test:
                    test_err = -preb_probs(f_log_probs,models_option,worddict,prepare_data,test,kf_test).mean()
                history_errs.append([valid, test_err])

                if udix == 0 or valid_err <= np.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    params = copy.copy(best_p)
                    params = unzip(tparams)
                    np.savez(saveto+'_bestll', history_errs=history_errs, **params)
                    bad_counter = 0

                if eidx > patience and len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience,0].min()
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break
                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

