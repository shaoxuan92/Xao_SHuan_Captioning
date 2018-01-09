# -*- coding:utf-8 -*-

xaoshuan_dataset = './data/xaoshuan_dataset'
if notexist(xaoshuan_dataset):
    mkdir(xaoshuan_dataset)
#
#xaoshuan_result = './data/xaoshuan_result.token'
#if is not exist(xaoshuan_result):
#   with open('result.token','w') as f:

image_list = list('flickr30k')[0:200]#从flickr30k随机选取200张图片

#把每幅图片都复制到新的文件夹，并把每幅图片的名字保持不变
for i in xrange(len(image_list)):
    xaoshuan_image = imread(i)
    with open('xaoshuan_dataset','w') as f:
        f.write(xaoshuan_image)

#把对应图片的caption放到xaoshuan_result.token里面
for i in xrange(len(image_list)):
    with open('result.token') as f:
        lines = re.match(f.readline,i)#找到与图片名字匹配的所有caption
        with open('xaoshuan_token_result') as g:
            g.write(lines,'wb')#把东西写入到里面



##以上就准备好了我的数据集和该数据集对应的caption
###########################################

#构造词典和（caption,feature)组成的pickle
#1.构造词典
a = CountVectors('xaoshuan_token_results')
dicionary = a.dictionary_()
with open('xaoshuan_dictionary') as f:
    f.write(dictionary,'wb')

#2.构造训练数据，提取每幅图片的特征
train,test,valid=split(xaoshuan_dataset)#把数据分成训练集，测试集和验证集
caffe_model = './caffemode'
caffe_deploy_protxt = './caffe_deploy_test'
CNN = caffe(caffe_deploy_protxt,caffe_model)
xaoshuan_features = CNN(train)
with open('xaoshuan_caption_feat_pair,pickle') as f:
    f.write(xaoshuan_features,captions)




