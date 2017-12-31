import cPickle as pkl
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