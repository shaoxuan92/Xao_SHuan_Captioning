#_*_coding:utf-8_*_
import numpy as np
import copy

class HomogeneousData():
    def __init__(self,data,batch_size=128,maxlen=None):
        self.batch_size = batch_size
        self.data = data
        self.maxlen=maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0] 
        self.feats = self.data[1]

        #find the unique lengths
        #就是把每句话的长度给统计下来，用到了split函数
        #默认为所有的空字符，包括空格、换行(\n)、制表符(\t)
        #Two small children in red shirts playing on a skateboard .
        self.lengths = [len(cc[0].split()) for cc in self.caps]
        self.len_unique = np.unique(self.lengths)
        #切掉 长度过长的 数值
        if self.maxlen:
            self.len_unique = [ ll for ll in self.len_unique if ll <= self.maxlen]

        #indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()

        for ll in self.len_unique:
            self.len_indices[ll] = np.where(self.lenghs == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        #这个的目的好像是打乱顺序
        #len_unique是[10,9,5,12]一类的东西
        #len_indices表示对应于各个长度的句子的标号如：
        #len_indices[10] = [1,200,1999,9999,33435353] 从而len_counts[10] = 5
        #len_indices[9] = [2,3553,16499] 从而len_counts[9] = 3
        #len_indices[5] = [454,1933,45921] 从而len_counts[5] = 3
        #len_indices[12] = [3,124,434353,124352] 从而len_counts[12] = 4
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0#这个是啥用呢？mark一下
            self.len_indices[ll] = np.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:#mod(0,12) = 0
            self.len_idx = np.mod(self.len_idx+1,len(self.len_unique))###从0开始，0，1，2，3.到第二次循环的时候（next)接着从4开始
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:##由于我们的句子长度有可能最低的是5.所以到第五的时候，执行下面一句话break
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration
        # batch size 我感觉貌似是根据句子长度相同的那些句子放在一起训练
        curr_batch_size = np.minimum(self.batch_size,self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]##貌似这个都是0
        # batch的index
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]##从0到batch_size

        # 对于每次循环取（next)出来的某个长度，len_indices_pos[5]或者len_indices_pos[9]都是从0编程curr_batch_size大小
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        #这个值貌似意义不是太大
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size
        caps = [self.caps[ii] for ii in curr_indices]#取出caps
    def __iter__(self):
        return self