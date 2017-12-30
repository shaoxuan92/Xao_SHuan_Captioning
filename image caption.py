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

def fflayer(tparams,state_below,options,prefix='ronv',activ='lambda x:tensor.tanh(x)',**kwargs):
#########state_below是一个3D矩阵，[n_step,Batch_size,dim_word]
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




###############################

##### Build_sampler###########

##############################
def build_sampler(tparams,options,use_noise,trng,ssampling=True):


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
          dim_word=100, n_words=1000
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
