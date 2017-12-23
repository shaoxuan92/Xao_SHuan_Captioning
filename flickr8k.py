import cPickle as pkl
def load_data(load_train=True,load_dev=True,load_test=True,path='./'):
    ###############

     #load data#

    ###############

    if load_train:
        with open(path+'flicker_8k_align.train.pkl','rb') as f:
    #here there is one question: two consecutive pkl.load(f) load two different things? like each column each load?
            train_cap = pkl.load(f)
            train_feat = pkl.load(f)