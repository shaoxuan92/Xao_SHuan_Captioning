import cPickle as pkl
import numpy as np
def load_data(load_train=True,load_dev=True,load_test=True,path='./'):
    ###############

     #load data#

    ###############
    ###############

     #load train data#

    ###############
    if load_train:
        with open(path+'flicker_8k_align.train.pkl','rb') as f:
    #there is one question: two consecutive pkl.load(f) load two different things? like each column each load?
            train_cap = pkl.load(f)
            train_feat = pkl.load(f)
        train=(train_cap,train_feat)
    else:
        train = None
    ###############

     #load dev data#

    ###############
    if load_dev:
        with open(path+'flicker_8k_align.dev.pkl','rb') as f:
            dev_cap = pkl.load(f)
            dev_feat = pkl.load(f)
        dev = (dev_cap,dev_feat)
    else:
        dev= None
    ###############

     #load test data#

    ###############
    if load_test:
        with open(path+'flicker_8k_align.test.pkl','rb') as f:
            test_cap = pkl.load(f)
            test_feat = pkl.load(f)
        test = (test_cap,test_feat)
    else:
        test = None
    ###############

     #load worddict#

    ###############
    with open(path+'dictionary.pkl','rb') as f:
        worddict = pkl.load(f)
    ###############

     #return data#

    ###############
    return train,dev,test,worddict


def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sequences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])

    lengths = [len(s) for s in seqs]
    y = np.zeros((len(feat_list),feat_list[0].shape[1])).astype('float32')

    for idx,ff in enumerate(feat_list):
        y[idx,:] = np.array(ff.todense())#todense 就是把稀疏矩阵转化为完整特征矩阵

    y = y.reshape([y.shape[0],14*14,512])

    n_samples = len(seqs)
    max_len = np.max(lengths) + 1

    x = np.zeros((max_len,n_samples)).astype('int64')
    x_mask = np.zeros((max_len,n_samples)).atype('float32')
    for idx,s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1

    return x,x_mask,y


