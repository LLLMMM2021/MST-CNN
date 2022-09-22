#!/usr/bin/env python
# coding: utf-8

# # 本文提出的模型

# ## 导入模块

# In[ ]:


import os
import sys
#reload(sys)
import imp
imp.reload(sys)
import importlib
importlib.reload(sys)
#import cPickle as pkl
import _pickle as pkl
import _pickle as cPickle
import pandas as pd
import numpy as np
import argparse
import matplotlib
import time, datetime
import re
matplotlib.use('Agg')
import matplotlib.pyplot as plt
seed = 1234
np.random.seed(seed)

from random import shuffle
from collections import Counter
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Dropout, concatenate, add
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer, Embedding, LSTM, Dense,Attention
from keras import initializers
from keras.layers import Input, Dense, merge
from keras.models import *
#np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
#from tsne import tsne
from sklearn.manifold import TSNE
from gensim.models import word2vec
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile


# ## 参数初始化

# In[ ]:


parser = argparse.ArgumentParser(description='embedconv training') #建立解析对象
parser.add_argument('-batchsize', dest='batchsize', type=int, default=128, help='size of one batch')#超参数
parser.add_argument('-init', dest='init', action='store_true', default=True, help='initialize vector')
parser.add_argument('-noinit', dest='init', action='store_false', help='no initialize')
parser.add_argument('-trainable', dest='trainable', action='store_true', default=True, help='embedding vectors trainable')
parser.add_argument('-notrainable', dest='trainable', action='store_false', help='not trainable')
parser.add_argument('-transform', dest='transform', action='store_true', default=True, help='transformation of the cost')
parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test step')
parser.add_argument('-filter', dest='filter', action='store_true', default=True, help='filter rare codes')
parser.add_argument('-isdays', dest='isdays', action='store_true', default=False, help='prediction of length of stay')
parser.add_argument('-dropout', dest='dpt', type=float, default=0.01, help='drop out rate')#超参数
parser.add_argument('-filtersize', dest='fz', type=int, default=3, help='filter region size')
parser.add_argument('-filternumber', dest='fn', type=int, default=100, help='filter numbers')#
parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate' )#超参数
parser.add_argument('-maxlen', dest='maxlen', type=int, default=17, help='max sequence length')
parser.add_argument('-dim', dest='dim', type=int, default=900, help='embedding vector length')#
parser.add_argument('-window', dest='window', type=int, default=10, help='word2vec window size')
args = parser.parse_args(args=[])

dropout = args.dpt
filter_size = args.fz
filter_number = args.fn
lr = args.lr
embedding_vector_length = args.dim
print('embedding vector length %d'%(embedding_vector_length))

if args.test:
    MID = 29
    SID = 48
else:
    i = datetime.datetime.now()
    MID = i.minute
    SID = i.second


# ## 步骤1:数据的清洗和导入

# In[ ]:


f = '/content/gdrive/MyDrive/15684_new_2.csv'
p = r'^[A-Z]'
pattern = re.compile(p) #将一个正则表达式编译成 Pattern 对象，可以利用 pattern 的一系列方法对文本进行匹配查找

def DataClean ():
    print('开始步骤1:数据的清洗和导入')
    data = pd.read_csv('/content/gdrive/MyDrive/15684_new_2.csv', encoding='utf-8')
    data = data[['target','Hospital_days', #预测目标值
                'Historical_days','Age','Hospital_times','Historical_times','time_interval','Date','Year','Month','Week','Historical_diagnoses',#患者特征
                'Diagnostic_code6']] #疾病特征
    data = data.dropna(subset=['Historical_days','Age','Hospital_times','Diagnostic_code6',]) #该函数主要用于滤除缺失数据
    cPickle.dump(data, open('/content/gdrive/MyDrive/dataclean.df', 'wb')) ##把data写入到后面链接文件中
    return data


# ## 步骤2:形成原始单词清单

# In[ ]:


def ToRawList(data):
    print('开始步骤2:形成原始单词清单')
    n_samples = len(data.index)
    # demographics, P27=days
    demographics = np.zeros((n_samples, 10))
    demographics[:, 0:1] = data[['Historical_days']].values 
    demographics[:, 1:2] = data[['Age']].values 
    demographics[:, 2:3] = data[['Hospital_times']].values
    demographics[:, 3:4] = data[['Historical_times']].values
    demographics[:, 4:5] = data[['Date']].values
    demographics[:, 5:6] = data[['Year']].values
    demographics[:, 6:7] = data[['Month']].values
    demographics[:, 7:8] = data[['Week']].values
    demographics[:, 8:9] = data[['Historical_diagnoses']].values
    #demographics[:, 9:10] = data[['time_interval']].values
    #diseases codes
    disease = data[['Diagnostic_code6']]
    disease = disease.fillna('')
    disease = disease.values
    disease = [[str(code).strip() for code in item if code != ''] for item in disease]
    #print(disease[:5])
    main_dis = data[['Diagnostic_code6']]
    main_dis = main_dis.fillna('')
    main_dis = main_dis.values
    l_disease = [len(item) for item in disease]
    l_disease = np.array(l_disease)
    C_disease = Counter(l_disease)
    #print("l_disease的最大长度:{}, 最小长度:{}".format(np.max(l_disease),np.min(l_disease))) 
    #print("l_disease的the 25% quarter:{}, the 75% quarter:{}, the mean:{}(+-){}".format(np.percentile(l_disease,25),np.percentile(l_disease,75),np.mean(l_disease),np.std(l_disease))) 
    l_disease = np.array(l_disease)

    seqs = data[['Historical_days','Age','Hospital_times','Historical_times','time_interval','Date','Year','Month','Week','Historical_diagnoses','Diagnostic_code6']]
    #seqs = data[['Diagnostic_code6']]
    seqs = seqs.fillna('')
    seqs = seqs.values
    seqs = [['#'.join(str(code).strip().split(' ')) for code in item if code != ''] for item in seqs]#replace the space with '#''
    seqs = [[str('0'+code) if len(code.split('.')[0])==1 else code for code in item] for item in seqs] #replace '3.90034' with the '03.90034'
    #print(seqs[:5])

    cost = data[['target']].values
    cost = np.asarray(cost, dtype=np.float32)
    days = data[['Hospital_days']].values
    days = np.asarray(days, dtype=np.float32)

    n_samples = len(seqs) #返回疾病编码的个数，包括相同值，即12548/15684
    #print(f,'#samples(seqs的长度):%d'%(n_samples)) # %d：以十进制形式输出带符号整数(正数不输出符号)

    main_dis = [[str(code) for code in item if code != ''] for item in main_dis] #输出每个单词
    C_maincodes = Counter([code for seq in main_dis for code in seq]) #计数，求每个单词出现的次数
    main_code = C_maincodes.keys() #形成单词字典
    n_dim = len(main_code) #共有242个单词
    #print("len(main_code):{}".format(n_dim))

    code2id = dict(zip(main_code, range(n_dim))) #zip：打包为元组的列表，相对于编号
    maincodemat = np.zeros((n_samples, n_dim), dtype=np.float32) #形成12548*242的0矩阵
    for i in range(n_samples):
        for code in main_dis[i]:
            if code in code2id:
                index = code2id[code]
                maincodemat[i,index] += 1
    #print("maincodemat.shape:{}".format(maincodemat.shape)) #形成单词矩阵maincodemat，出现，则该行为1
    Data = (seqs, cost, days, demographics,disease)
    pkl.dump(Data, open('/content/gdrive/MyDrive/Data.df','wb'))
    return seqs, cost, days, demographics, disease, main_dis


# ## 步骤3:给单词编号,形成单词-编号的字典

# In[ ]:


#步骤3:给单词编号
def token_to_index(seqs):
    print('开始步骤3:给单词编号,形成单词-编号的字典')
    C_codes = Counter([code for seq in seqs for code in seq])
    code_index = {}
    for idx, item in enumerate(C_codes.keys()): #enumerate：同时列出数据、数据下标
        code_index[item] = idx + 1
    #print("the unique codes: {}".format(len(C_codes.keys())))
    return code_index   #输入code_index，输出：{'E10.901': 1,...}


# ## 步骤4:词嵌入

# In[ ]:


def get_index_embedding(code_index={}, level=0):
    print("开始第4步")
    index_embedding = {}
    cnt = 0
    dim = 900
    window = 10    #glove文件包含各种大小的文本编码向量:50维、100维、200维、300维
    #model = KeyedVectors.load_word2vec_format(word_vector_path, binary=True)
    model = KeyedVectors.load('/content/gdrive/MyDrive/newlevel%d_word2vec_dim%d_window%d.model' %(level, dim, window))
   
    for code, index in code_index.items():  # items(): 返回可遍历的(键, 值) 元组数组
        #print(re.findall(pattern, code))
        if len(re.findall(pattern, code))== 1:
            if level == 0:
                newcode = code
            if level == 1:
                newcode = code[0:3]
            if level == 2:
                newcode = code[0:5]
            if level == 3:
                newcode = code[0:6]
        else:
            if level == 0:
                newcode = code
            if level == 1:
                newcode = code[0:3]
            if level == 2:
                newcode = code[0:4]
            if level == 3:
                newcode = code[0:5]

        if newcode in model:
            index_embedding[index] = model[newcode]
        else:
            cnt = cnt + 1
            index_embedding[index] = np.random.uniform(-0.25,0.25,embedding_vector_length)
    return index_embedding #即619个疾病代码，每个代码映射成了100维的向量

def get_trained_embedding(index_embedding=None):
    print("开始get_trained_embedding")
    index_sorted = sorted(index_embedding.items()) # sorted by index starting from 1 ;items(): 返回可遍历的(键, 值) 元组数组
    trained_embedding = [t[1] for t in index_sorted] #len(trained_embedding):424 type:<class 'list'>
    embedding_vector_length = args.dim
    zeros = np.random.uniform(-0.25,0.25,embedding_vector_length) #一维的数组  zeros.shape:(424,)
    trained_embedding = np.vstack((zeros, trained_embedding)) #它是垂直（按照行顺序）的把数组给堆叠起来。
    trained_embedding = np.array(trained_embedding) #trained_embedding.shape:(101, 424);trained_embedding.维度:2
    return trained_embedding

#步骤5:单尺度编码 E10.901 C12.783
def embedding_encoder(idseqs, index_embedding):
    print('开始步骤51/3:单尺度编码')
    n_samples = len(idseqs)
    dim = args.dim
    mat = np.zeros((n_samples, dim), np.float32)
    for i in range(n_samples):
        for codeid in idseqs[i]:
            if codeid in index_embedding:
                code_embedding = np.array(index_embedding[codeid])
            else:
                code_embedding = np.random.uniform(-0.25, 0.25, dim)
            mat[i] += code_embedding
    mat = np.array(mat)
    return mat

#多尺度编码 E10.901 C12.783     E10.90 C12.78  E10.9      6-5-4-3
def multichannel_embedding_encoder(idseqs, index_embedding,index_embedding1,index_embedding2,index_embedding3):
    print('开始步骤52/3:开始多尺度编码...')
    n_samples = len(idseqs)
    dim = args.dim
    mat = np.zeros((n_samples, dim), np.float32)
    for i in range(n_samples):
        for codeid in idseqs[i]:
            code_embedding = np.asarray([0]*dim, np.float32)
            if codeid in index_embedding:
                code_embedding += np.array(index_embedding[codeid])
            if codeid in index_embedding1:
                code_embedding += np.array(index_embedding1[codeid])
            if codeid in index_embedding2:
                code_embedding += np.array(index_embedding2[codeid])
            if codeid in index_embedding3:
                code_embedding += np.array(index_embedding3[codeid])
            mat[i] += code_embedding
    mat = np.array(mat) 
    return mat

#独热码
def one_hot_encoder(seqsind, nb_words):
    print('开始步骤53/3:开始独热编码one_hot encoding...')
    n_samples = len(seqsind)
    n_dim = nb_words
    mat = np.zeros((n_samples, n_dim), dtype=np.float32)
    for i in range(n_samples):
        for codeid in seqsind[i]:
            if codeid !=0:
                index = codeid - 1 # for the seqid starts from 1
                mat[i,index] = mat[i,index] + 1
    return mat

#步骤6:
def svd(seqsind, dim=args.dim): #600 奇异值分解(SVD)
    print("开始61/3:开始svd")
    mat = one_hot_encoder(seqsind, nb_words)
    #print("mat.shape:{}".format(mat.shape))
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dim, random_state=42)
    print('SVD......') 
    svd.fit(mat)
    res = svd.transform(mat)
    print('SVD done!') 
    return res

def glove(level, dim=args.dim, window=10): 
    print("开始62/3:开始glove")
    vectors = '/content/gdrive/MyDrive/weighted1_vectors%d'%(level)
    model = KeyedVectors.load_word2vec_format(vectors, binary=False)
    name = 'newlevel%d_weighted1_glove_dim%d_window%d.model'%(level, dim, window)
    model.save('/content/gdrive/MyDrive/cnn_model/'+name)
    return model

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


# ## 步骤5:加载数据

# In[ ]:


#步骤7:加载数据
def load_data(X,demographics, y, onehot_mat):
    print('开始步骤7:划分训练集、验证集、测试集...')
    print("y:",y)
    #y=np.log(y)
    indices = np.arange(n_seqs)
    np.random.shuffle(indices) #打乱
    X = X[indices]
    demographics = demographics[indices]
    y = y[indices]
    #onehot_mat = onehot_mat[indices]

    n_tr = int(n_seqs * 0.85)
    n_va = int(n_seqs * 0.05)
    n_te = n_seqs - n_tr - n_va
    X_train = X[:n_tr]
    demographics_train = demographics[:n_tr]
    y_train = y[:n_tr]
    #onehot_mat_train = onehot_mat[:n_tr]

    X_valid = X[n_tr:n_tr+n_va]
    demographics_valid = demographics[n_tr:n_tr+n_va]
    y_valid = y[n_tr:n_tr+n_va]
    #onehot_mat_valid = onehot_mat[n_tr:n_tr+n_va]

    X_test = X[-n_te:]
    demographics_test = demographics[-n_te:]
    y_test = y[-n_te:]
    #onehot_mat_test = onehot_mat[-n_te:]

    #print np.max(y_test),np.min(y_test)
    return X_train, X_test, X_valid, y_train, y_test, y_valid, demographics_train, demographics_test, demographics_valid#, onehot_mat_train,onehot_mat_test,onehot_mat_valid


def filter_test(X_test, index_code, threshold=2):
    print('开始步骤8:选择罕见集样本用于测试集...，这一步其实是将所有数据15684都用于了测试集')
    comm_inds = []
    for ind, seq in enumerate(X_test):
        cnt = 0
        for index in seq:
            if index in index_code:
                code = index_code[index]
            else:
                code = 'none'
        if cnt < threshold:
            comm_inds.append(int(ind))
    print('comm_inds:%d'%(len(comm_inds)))
    comm_inds = np.asarray(comm_inds)
    return  comm_inds

def evaluation(y_test, y_pred):
    print('y_test:{}'.format(y_test.shape))
    print('y_pred:{}'.format(y_pred.shape))
    print(y_pred.dtype) #float32
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MSE:{}, RMSE:{}, MAE:{}, R2:{}'.format(mse, rmse, mae,r2)) 
    return r2, rmse


# ## 可视化

# In[ ]:


#步骤8:预测、可视化
def cross_validation():
    print("开始5折交叉验证")
    estimator = KerasRegressor(build_fn=cnn_model, epochs=100, batch_size=args.batchsize, verbose=0)
    kfold = KFold(n_splits=5, random_state=seed)
    res = cross_val_score(estimator, X, y, cv=kfold)
    print( 'cross validation for cnn model ......')
    print('cnn_model: MSE %.2f (+-)%.2f'%(res.mean(), res.std())) 
#cross_validation()
def daysplots(y_test, y_pred, r2, rmse, name):
    print("开始步骤91/2:预测可视化-住院天数")
    fig, ax = plt.subplots()
    #ax.set_xlim(0, 35)
    #ax.set_ylim(0, 35)
    #ax.plot(y_test.squeeze(), y_test.squeeze(), s=10, marker='.',c='r')
    ax.scatter(y_test.squeeze(), y_pred, s=10,marker='.', c='b')
    ax.text(28,30,'R2:{:.4f}'.format(r2))
    ax.text(28,32,'RMSE:{:.2f}'.format(rmse))
    ax.set_xlabel('True length of stay (days)')
    ax.set_ylabel('predicted length of stay (days)')
    fig.savefig('/content/gdrive/MyDrive/%s_days.pdf'%(name))

def costplots(y_test, y_pred, r2, rmse, name):
    print("开始步骤92/2:预测可视化-医疗费用")
    fig, ax = plt.subplots()
    #ax.set_xlim(0, 12000)
    #ax.set_ylim(0, 12000)
    ax.scatter(y_test.squeeze(), y_pred, s=10,marker='o',c='b') #c='b'
   # ax.text('R2:{:.4f}'.format(r2)) #8000,10000,
    #ax.text('RMSE:{:.2f}'.format(rmse)) #8000,11000,
    ax.set_xlabel('Series')
    ax.set_ylabel('Medical cost')
    fig.savefig('/content/gdrive/MyDrive/%s_cost.png'%(name),dpi=500,bbox_inches = 'tight')

    #画散点图
    plt.figure()#figsize=(15,5)
    plt.plot(y_test,'rs',label='True value')
    plt.plot(y_pred,'go',label='Predict value')
    #plt.title('Predicted and true values of medical costs',fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Series',fontsize=13)
    plt.ylabel('Medical Cost',fontsize=13)
    plt.legend(loc=1,fontsize=10,frameon=True)
    plt.savefig('/content/gdrive/MyDrive/%s_散点图.png'%(name),dpi=500,bbox_inches = 'tight')

    fig, ax = plt.subplots()
    plt.plot(y_test, linewidth = 1,linestyle='-',label='True Value') #marker='*'
    plt.plot(y_pred, linewidth = 1,linestyle='-',label='Predicted Value') #marker='^',
    #ax.text('R2:{:.4f}'.format(r2)) #8000,10000,
    #ax.text('RMSE:{:.2f}'.format(rmse)) #8000,11000,
    plt.xlabel('Series',fontsize=13)
    plt.ylabel('Medical Cost',fontsize=13)
    plt.xlabel('Series',fontsize=13)
    plt.ylabel('Medical Cost',fontsize=13)
    fig.savefig('/content/gdrive/MyDrive/%s_曲线图.pdf'%(name))


# In[ ]:


def extract_patientvec(model,modelpath, disease,  demographics):
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath),by_name=False)
    sub_model1 = Model(inputs=model.inputs, outputs=model.get_layer('F1').output)#错误代码修改
    sub_model2 = Model(inputs=model.inputs, outputs=model.get_layer('F2').output)#错误代码修改
    sub_model3 = Model(inputs=model.inputs, outputs=model.get_layer('F3').output)#错误代码修改
    sub_model4 = Model(inputs=model.inputs, outputs=model.get_layer('F4').output)#错误代码修改
    sub_model5 = Model(inputs=model.inputs, outputs=model.get_layer('F5').output)#错误代码修改

    patientvecs3 = sub_model3.predict([disease,  demographics], verbose=1)
    modelname='sub_model3'
    print(patientvecs3.shape)
    pkl.dump(patientvecs3, open('/content/gdrive/MyDrive/cnn_model/patientvec_%s'%(modelname),'wb'))

    patientvecs4 = sub_model4.predict([disease, demographics], verbose=1)
    modelname='sub_model4'
    print(patientvecs4.shape)
    pkl.dump(patientvecs4, open('/content/gdrive/MyDrive/cnn_model/patientvec_%s'%(modelname),'wb'))

    patientvecs5 = sub_model5.predict([disease,  demographics], verbose=1)
    modelname='sub_model5'
    print(patientvecs5.shape)
    pkl.dump(patientvecs5, open('/content/gdrive/MyDrive/cnn_model/patientvec_%s'%(modelname),'wb'))


# ## 注意力机制

# In[ ]:


#CBAM
#导入注意力机制所需要的库
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Dense, multiply, Permute, Concatenate, Conv1D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
#1、通道注意力机制  具体来说他就是是对应一个全局平均池化的操作。将一个c通道，hxw的特征图，压成c通道1x1 可以看作一个C维向量
#SENet采用一个小型的子网络，获得一组权重，进而将这组权重与各个通道的特征分别相乘，以调整各个通道特征的大小。
#这个过程，就可以认为是在施加不同大小的注意力在各个特征通道上。
#在SENet中，获得权重的具体路径是，“全局池化→全连接层→ReLU函数→全连接层→Sigmoid函数”。
def channel_attention(input_feature, ratio=2):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1 #mage_data_format返回图像的维度顺序
    print("input_feature.shape:{}".format(input_feature.shape))
    print("channel_axis:{}".format(channel_axis)) #channel_axis:-1
    channel = input_feature.shape[channel_axis] #这个作用就是找到通道所在的位置 shape[0]读取矩阵第一维度的长度 行数 ，shape[1]:列数，每行有几个元素
    print("channel:{}".format(channel)) #channel:1300

    shared_layer_one = Dense(channel//ratio, #进行全连接，得到C/r维的向量
                             kernel_initializer='he_normal', #He正态分布初始化方法 
                             activation = 'relu', #prelu带参数的ReLU
                             use_bias=True, #在进行Relu激活，再对进行一次全连接，将C/r维的向量变回C维向量，再进行sigmoid激活（使得数值位于0-1之间），这便是得到了权重矩阵。
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling1D()(input_feature) #输入（samples,steps,features）的3D张量，则输出形如（samples,features）的2D张量
    avg_pool = Reshape((1,channel))(avg_pool) #如果是AveragePooling1D,则输入（samples,steps,features）的3D张量，则输出形如（samples,downsampled_steps,features）的3D张量
    assert avg_pool.shape[1:] == (1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,channel//ratio) #C/r（r为减少率）
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,channel) #第二层神经元个数为 C，这个两层的神经网络是共享的。
    
    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool]) # 将MLP输出的特征进行基于element-wise的加和操作，再经过sigmoid激活操作，
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature) #Permute：可以同时多次交换tensor的维度
    
    return multiply([input_feature, cbam_feature]) #最后，将M_c和输入特征图F做element-wise乘法操作，生成Spatial attention模块需要的输入特征

#2、空间注意力机制
def spatial_attention(input_feature):
    kernel_size = 3 # #卷积核的尺寸 #实际上卷积大小为kernel_size* in_channels输入信号的通道,在文本分类中，即为词向量的维度
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature) #将tensor的维度换位。
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(cbam_feature) # 首先做一个基于channel的global max pooling 
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cbam_feature) # 和global average pooling，得到两个H×W×1 的特征图
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=2)([avg_pool, max_pool]) #然后将这2个特征图基于channel 做concat操作（通道拼接）
    assert concat.shape[-1] == 2
    cbam_feature = Conv1D(filters = 1, #filters 即为输出的维度       #然后经过一个7×7卷积，降维为1个channel
                    kernel_size=kernel_size, #卷积核的尺寸
                    activation = 'hard_sigmoid',  #再经过sigmoid生成spatial attention feature
                    strides=1, #卷积步长
                    padding='same', #输入的每一条边补充0的层数
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) #如果bias=True，添加偏置
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3,1,2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature]) #最后将该feature和该模块的输入feature做乘法，得到最终生成的特征。 

#3、构建CBAM
def cbam_block(cbam_feature,ratio=2):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )
    return cbam_feature


# ## 多尺度划分——注意力模型

# In[ ]:


#第四个模型：多尺度划分——注意力模型
# se注意力机制 现有注意力模块的另一个重要影响因素：权值生成方法。现有注意力往往采用额外的子网络生成注意力权值，比如SE的GAP+FC+ReLU+FC+Sigmoid。
def se_block(inputs, ratio=2):  # ratio代表第一个全连接层下降通道数的系数
    in_channel = inputs.shape[-1]# 获取输入特征图的通道数
    print('通道数:{}'.format(in_channel))
    x = layers.GlobalAveragePooling1D()(inputs) # 全局平均池化[h,w,c]==>[None,c] #压缩：得到当前Feature Map的全局压缩特征量
    x = layers.Reshape(target_shape=(1,1,in_channel))(x) # [None,c]==>[1,1,c]
    x = layers.Dense(in_channel//ratio)(x)  # 全连接下降通道数 # [1,1,c]==>[1,1,c/4] #激发：通过两层全连接的bottleneck结构得到Feature Map中每个通道的权值
    x = tf.nn.relu(x) # relu激活
    x = layers.Dense(in_channel)(x)  # 全连接上升通道数 # [1,1,c/4]==>[1,1,c]
    x = tf.nn.sigmoid(x) # sigmoid激活，权重归一化
    print("sigmoid激活，权重归一化:",x)
    outputs = layers.multiply([inputs, x])  # 归一化权重和原输入特征图逐通道相乘 # [h,w,c]*[1,1,c]==>[h,w,c] #激发：并将加权后的Feature Map作为下一层网络的输入。
    return outputs  
#eca注意力机制：相比于se模块，实现了适当的跨通道交互而不是像全连接层一样全通道交互。（用一维卷积替换了全连接层）
#通过执行卷积核大小为k的一维卷积来生成通道权重，其中k通过通道维度C的映射自适应地确定。
import math
def eca_block(inputs, b=1, gama=2):
    in_channel = inputs.shape[-1] # 输入特征图的通道数
    print('通道数:{}'.format(in_channel))
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama)) # 根据公式计算自适应卷积核大小
    if kernel_size % 2:              # 如果卷积核大小是偶数，就使用它
        kernel_size = kernel_size
    else:                            # 如果卷积核大小是奇数就变成偶数
        kernel_size = kernel_size + 1
    x = layers.GlobalAveragePooling1D()(inputs) # [h,w,c]==>[None,c] 全局平均池化
    x = layers.Reshape(target_shape=(in_channel, 1))(x) # [None,c]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x) # [c,1]==>[c,1]
    x = tf.nn.sigmoid(x) # sigmoid激活
    x = layers.Reshape((1,1,in_channel))(x) # [c,1]==>[1,1,c]
    outputs = layers.multiply([inputs, x]) # 结果和输入相乘
    return outputs
##
def prelu(_x): #name=None
    """parametric ReLU activation"""
    print("_x:",_x.get_shape(),"_x.get_shape()[-1]:",_x.get_shape()[-1])
    _alpha = tf.compat.v1.get_variable("prelu", #name +
                             shape=_x.get_shape()[-1], #get_shape():得到张量（数组）的维度
                             dtype=_x.dtype,
                             initializer=tf.constant_initializer(0.001))
    pos = tf.nn.relu(_x)
    neg = _alpha * (_x - tf.abs(_x)) * 0.5

    return pos + neg


import tensorflow as tf
from keras.layers import BatchNormalization

def ATTENTION_multi_channel_split_model(demgras_dim):
    dis_in = Input(shape=(DIS_MAX_LEN, ), dtype='float32')
    dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(dis_in)
    dis_embedding1level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix1],
                                trainable=args.trainable)(dis_in)
    dis_embedding2level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix2],
                                trainable=args.trainable)(dis_in)
    dis_embedding3level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix3],
                                trainable=args.trainable)(dis_in)
    #bn = BatchNormalization() #BN层就是为了让让每一层的值在一个有效范围内传递下去。1、加快网络的训练和收敛的速度 2、控制梯度爆炸防止梯度消失 3、防止过拟合
    #fusion_vector = concatenate([dis_embedding0level, dis_embedding1level, dis_embedding2level, dis_embedding3level])
    conv_result = []
    for i in range(3):
        channel_result = []
        #lstm = LSTM(units=1000, recurrent_activation='leaky_relu', dropout=0.1) 
        #lstm_out = lstm(fusion_vector)
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(dis_embedding0level)
        conv1 = conv_layer(dis_embedding1level)
        conv2 = conv_layer(dis_embedding2level)
        conv3 = conv_layer(dis_embedding3level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        pooling1 = GlobalMaxPooling1D()(conv1)
        pooling2 = GlobalMaxPooling1D()(conv2)
        pooling3 = GlobalMaxPooling1D()(conv3)
        channel_result.append(pooling0)
        channel_result.append(pooling1)
        channel_result.append(pooling2)
        channel_result.append(pooling3)
        allchannel = add(channel_result)
        conv_result.append(allchannel)
        #bn = BatchNormalization()
        #conv_result.append(lstm_out)

    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    print("demgras_dim:{}".format(demgras_dim)) #demgras_dim:11
    print("demgras_in.shape:{}".format(demgras_in.shape))#(None, 11) sigmoid
    dense_demgras = Dense(1000, activation='sigmoid')(demgras_in) #它将任意的值转换到 [0,1] 之间 BN层是将数据转换为均值为0，方差为1的正态分布
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result) #numpy.concatenate函数 主要作用:沿现有的某个轴对一系列数组进行拼接。
    
    merge_out = tf.expand_dims(merge_out, axis=0)
    
    x = se_block(merge_out)  # 接收SE返回值 #可行 但是平均误差突然间猛增 动荡 这个网络不错诶 虽然前期增加，但是后期整体趋势在下降，虽然偶尔动荡
    x=tf.squeeze(x,[0, 1])

    #x = eca_block(merge_out)  # 接收ECA输出结果 这个网络类似于上面这个网络 等会可以试试 和SENet相比大大减少了参数量，参数量等于一维卷积的kernel_size的大小
    #x=tf.squeeze(x,[0, 1])
    
    #cbam = cbam_block(merge_out) #尝试把注意力机制放在这
    #print("cbam:{}".format(cbam.shape))
    #x=tf.squeeze(cbam,[0])
    #print(x.shape)
    
    dense_out = Dense(1000, activation='relu', name='F1')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(500, activation='relu', name='F2')(dpt) 
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu', name='F3')(dpt) #RReLU中的aji是一个在一个给定的范围内随机抽取的值，这个值在测试环节就会固定下来。
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(50, activation='relu', name='F4')(dpt) #Leaky ReLU中的ai是固定的
    dpt = Dropout(dropout)(dense_out)
    #activation_function = keras.layers.advanced_activations.PReLU(init='zero', weights=None) 
    dense_out = Dense(2, activation='leaky_relu', name='F5')(dpt) #leaky_relu elu PReLU #PReLU中的ai是根据数据变化的#
    #dense_out = prelu(dense_out)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([dis_in,demgras_in], mode_out) 
    #rmsprop = optimizers.rmsprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0) 均方根传播(RMSProp) ;根据最近的权重梯度的平均值(例如变化的速度)来调整;这意味着该算法在线上和非平稳问题上表现良好(如:噪声)。
    Adam=tf.keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
    #rmsprop tf.keras.losses.Huber()
    model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mae']) #tf.keras.losses.Huber()
    print(model.summary())
    return model

def train_ATTENTION_multi_channel_split_model(model, modelpath, X1_train,  demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1)#patience：能够容忍多少个epoch内都没有improvement。verbose：日志显示 verbose = 1 为输出进度条记录

    print('Training the ATTENTION_merge model....') 
    history =model.fit([X1_train,  demographics_train], y_train, epochs=200, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper], # 尝试关闭早停
              verbose=1)
    print("4、开始画图：损失")
    from matplotlib import rcParams
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
               'size': 15,}
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss',fontsize=15)
    plt.xlabel('Epoch',font2,verticalalignment='top') # fontsize=14, 
    plt.ylabel('Loss',font2, horizontalalignment='center') #fontsize=14, 
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc=1,fontsize=10,frameon=True)
    plt.savefig('/content/gdrive/MyDrive/LOSS.png',dpi=500,bbox_inches = 'tight')
 
def test_ATTENTION_multi_channel_split_model(model, modelpath, X1_test,  demographics_test, y_test,index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test, demographics_test], batch_size=args.batchsize, verbose=1)
    print("y_pred:{}".format(y_pred.shape))
    r2,rmse = evaluation(y_test, y_pred)
    print("r2:",r2,"rmse:",rmse)
    #画图1
    name = 'ATTENTION_MG_split%d'%(index)
    costplots(y_test, y_pred, r2, rmse, name)
    #画图2
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)
    #画图3
    print("!!开始画图：预测值与真实值")
    print("y_test:",y_test.shape)
    print("y_pred:",y_pred.shape)
    print("y_pred.ndim:",y_pred.ndim)
    print("y_test.ndim:",y_test.ndim)
    from matplotlib import rcParams
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
               'size': 15,}

    plt.figure()
    plt.plot(list(range(1569)),y_test,label='True Value')
    plt.plot(list(range(1569)),y_pred,label='Predicted Value')
    b=np.squeeze(y_test)
    c=np.squeeze(y_pred)
    plt.fill_between(list(range(1569)),b,c,color='g',alpha=.25)
    plt.title('Predicted and true values of medical costs',fontsize=13)
    plt.xlabel('Series',fontsize=13, verticalalignment='top') #
    plt.ylabel('Medical cost',fontsize=13,horizontalalignment='center') #
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc=1,fontsize=10,frameon=True)
    plt.savefig('/content/gdrive/MyDrive/Attention_曲线图.png',dpi=500,bbox_inches = 'tight')
    #画散点图
    plt.figure()#figsize=(15,5)
    plt.plot(y_test,'rs',label='True value')
    plt.plot(y_pred,'go',label='Predict value')
    plt.title('Predicted and true values of medical costs',fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Series',fontsize=13,)
    plt.ylabel('Medical Cost',fontsize=13)
    plt.legend(loc=1,fontsize=10,frameon=True)
    plt.savefig('/content/gdrive/MyDrive/Attention_散点图.png',dpi=500,bbox_inches = 'tight')
    #画：测试集的预测损失
    plt.figure()
    a=y_test-y_pred
    plt.plot(a, label='Loss')
    plt.title('Predicted loss of the proposed model',fontsize=13)
    plt.xlabel('Series',fontsize=10, verticalalignment='top') #
    plt.ylabel('Loss value',fontsize=10,horizontalalignment='center') #
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('/content/gdrive/MyDrive/Predicted_LOSS.png',dpi=500,bbox_inches = 'tight')


# # 开始预测

# ## 准备数据

# In[ ]:



path = 'weighted_%s:%s_%sdays_%sinit_%strainable_%fdpt_%flr_%dfz_%dfn_%dmaxlen_%ddim_%sfilter_%stransform %sbatch_%swindow'%(MID, SID, args.isdays, args.init, args.trainable, args.dpt, args.lr, args.fz, args.fn, args.maxlen, args.dim, args.filter, args.transform,args.batchsize,args.window)

if os.path.isfile('/content/gdrive/MyDrive/dataclean.df'):
    data = pkl.load(open('/content/gdrive/MyDrive/dataclean.df','rb'))
else:
    data = DataClean()
rawseqs, cost, days, demographics, disease, main_dis = ToRawList(data) #rawseqs就是seqs
#print("rawseqs:",rawseqs)

seqs1levels = []
seqs2levels = []
seqs3levels = []
for seqs in rawseqs:
    levels1 = []
    levels2 = []
    levels3 = []
    for code in seqs:
        levels1.append(code[0:3])
        if len(re.findall(pattern, code))== 1:
            levels2.append(code[0:5])
            levels3.append(code[0:6])
        else:
            levels2.append(code[0:4])
            levels3.append(code[0:5])
    seqs1levels.append(levels1)
    seqs2levels.append(levels2)
    seqs3levels.append(levels3)

print('开始将疾病诊断代码转换为索引')
code_index = token_to_index(rawseqs)# transform code into index 将代码转换为索引
index_code = dict([(kv[1], kv[0]) for kv in code_index.items()])# validate the code_index 验证代码_索引
#print [[index_code[index] for index in item ] for item in seqs[0:2]]

idseqs = [[code_index[code] for code in item] for item in rawseqs]
iddis = [[code_index[code] for code in item] for item in disease]
idmain_dis = [[code_index[code] for code in item] for item in main_dis]
#print(main_dis)
#"""
print('开始形成词嵌入向量')
print(disease)
print(disease[1:]) #list 15684
word2vec_model(disease,0)
index_embedding = get_index_embedding(code_index, 0)
embedding_matrix0 = get_trained_embedding(index_embedding)
word2vec_model(disease, 1)
index_embedding1 = get_index_embedding(code_index, 1)
embedding_matrix1 = get_trained_embedding(index_embedding1)
word2vec_model(disease, 2)
index_embedding2 = get_index_embedding(code_index, 2)
embedding_matrix2 = get_trained_embedding(index_embedding2)
word2vec_model(disease, 3)
index_embedding3 = get_index_embedding(code_index, 3)
embedding_matrix3 = get_trained_embedding(index_embedding3)

main_code = []
for item in idmain_dis:
    for ind in item:
        main_code.append(embedding_matrix0[1][ind])

maincodemat = np.array(main_code)

nb_words = len(index_code) # code_index starting from 1
max_features = nb_words + 1 
n_seqs = len(idseqs)

MAX_LEN = args.maxlen      # the max len is 21 #这里没有治疗编码，所以最大长度为以记录为单位
seqs = pad_sequences(idseqs, maxlen=MAX_LEN)
DIS_MAX_LEN = 17  #其实这里需要改成以记录为单位才对 DIS_MAX_LEN = 11 SUR_MAX_LEN = 10 maxlen:21
disease = pad_sequences(iddis, maxlen=DIS_MAX_LEN)


X = np.array(seqs)
X1 = np.array(disease)
print("X(seqs):",X.shape,"X1(disease):",X1.shape)
print("seqs:",seqs.shape)


"""
if args.test:
    modelpath = 'MG_merge' + path
    demgras_dim = 3
    model= multi_channel_split_model(demgras_dim)
    extract_patientvec(model, modelpath, disease, surgery, demographics)
"""
#demographics = np.hstack((demographics, maincodemat))
if args.isdays:
    y = np.asarray(days, dtype='float32')
else:
    y = np.asarray(cost,dtype='float32')

if args.transform:
    print("np.log")
    y = np.log(y)
print(X, demographics)

comm_inds= filter_test(X, index_code, 4)
comm_X = X[comm_inds]
comm_X1 = X1[comm_inds]
comm_demographics = demographics[comm_inds]
comm_y = y[comm_inds]
print("comm_X:",comm_X,"comm_X1:",comm_X1,"comm_y:",comm_y)

print('Spliting train, test parts...')
X_train, X_test, X1_train, X1_test, y_train, y_test, demographics_train, demographics_test = train_test_split(comm_X, comm_X1,comm_y, comm_demographics, test_size=0.1, random_state=90)
print('X_train:{}'.format(X_train.shape))
print('X_test:{}'.format(X_test.shape))
print('X1_train:{}'.format(X1_train.shape))
print('X1_test:{}'.format(X1_test.shape))
print('y_train:{}'.format(y_train.shape))
print('y_test:{}'.format(y_test.shape))
print('demographics_train:{}'.format(demographics_train.shape))
print('demographics_test:{}'.format(demographics_test.shape))

#y_train=np.log(y_train)
#y_test=np.log(y_test)
print("y_test:",y_test)
  
demgras_dim = demographics.shape[1] #图像的水平尺寸 
print("demgras_dim:{}".format(demgras_dim)) #demgras_dim:10


# ## 开始预测

# In[ ]:


#=============================== 第四部分 开始预测 =====================================================
r2_CNN_val=[]
r2_CNN_test=[]
for i in range(1):
    print('第%s次训练'%i)
  
    modelpath = '9月1日(上午)_'+'第%s次训练_'%i+'MG_merge' + path # %i
    #定义模型
    model= ATTENTION_multi_channel_split_model(demgras_dim) 
    #划分数据
    X1_train_1,X1_val_1, demographics_train_1,demographics_val_1,y_train_1,y_val_1=train_test_split( X1_train, demographics_train, y_train, test_size=0.1, random_state=90) 
    if not args.test:
        train_ATTENTION_multi_channel_split_model(model, modelpath, X1_train_1, demographics_train_1, y_train_1)
    extract_patientvec(model, modelpath, disease, demographics)
    #拟合、预测
    y_CNN_hat = model.predict([X1_val_1,demographics_val_1])

    r2,rmse = evaluation(y_val_1, y_CNN_hat) #验证集的拟合优度
    R21=r2_score(y_val_1, y_CNN_hat)
    print(i,R21)
    r2_CNN_val.append(R21)

    test_ATTENTION_multi_channel_split_model(model, modelpath, X1_test, demographics_test,y_test,3)
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_CNN_pred = model.predict([X1_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r22,rmse = evaluation(y_test, y_CNN_pred)
    print("r2:",r22,"rmse:",rmse)
    r2_CNN_test.append(r22)


# ## 预测结果

# In[ ]:


r2_CNN_test


# # 对比模型一

# In[ ]:


#单尺度模型
####这里这里这里！！！！！！！！！！
from tensorflow.keras import optimizers # 错误代码1的改正方案
#模型1:单尺度融合模型
def single_channel_merge_model(demgras_dim):
    print('开始模型1:单尺度融合模型')
    codes_in = Input(shape=(MAX_LEN, ), dtype='float32') #MAX_len 输入句子的最大长度17
    if args.init:
        print('initialize embedding layer with pre-training vectors')
        print('依据预训练向量 初始化嵌入层')
        print('embedding layers trainalble %s' % args.trainable) 
        embedding0level = Embedding(output_dim=embedding_vector_length,  #424
                                input_dim=max_features, 
                                input_length=MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(codes_in)
    else:
        print('one hot embedding with random initialization...') 
        embedding0level = Embedding(output_dim=embedding_vector_length,
                                    input_dim=max_features,
                                    embeddings_initializer='random_uniform',
                                    input_length=MAX_LEN)(codes_in)
    conv_result = []
    for i in range(3):
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(embedding0level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        conv_result.append(pooling0)

    print('添加人口统计信息，融合')
    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    dense_demgras = Dense(3, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result)
    dense_out = Dense(500, activation='relu')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([codes_in, demgras_in], mode_out)
    #rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0) 出错代码1
    #rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0) #weight_decay的作用是正则化
    rmsprop = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae']) #MSE均方误差：真实值-预测值 然后平方之后求和平均 #MAE：平均绝对误差
    print(model.summary())
    return model

#模型2:单尺度划分模型
def single_channel_split_model(demgras_dim):
    dis_in = Input(shape=(DIS_MAX_LEN, ), dtype='float32')
    if args.init:
        print('initialize embedding layer with pre-training vectors')
        print('依据预训练向量，初始化嵌入层...')
        print('embedding layers trainalble %s' % args.trainable) 
        dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                        input_dim=max_features,
                                        input_length=DIS_MAX_LEN,
                                        weights=[embedding_matrix0],
                                        trainable=args.trainable)(dis_in)

    else:
        print('one hot embedding with random initialization...')
        print('随机初始化,独热嵌入层...')
        dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                embeddings_initializer='random_uniform')(dis_in)

    conv_result = []
    for i in range(3):
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(dis_embedding0level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        conv_result.append(pooling0)

    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    dense_demgras = Dense(3, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result)
    dense_out = Dense(1000, activation='relu')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(500, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([dis_in,  demgras_in], mode_out) #sur_in,
    rmsprop = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])
    print(model.summary())
    return model

#第三个模型：多尺度划分模型
def multi_channel_split_model(demgras_dim):
    dis_in = Input(shape=(DIS_MAX_LEN, ), dtype='float32')
    dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(dis_in)
    dis_embedding1level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix1],
                                trainable=args.trainable)(dis_in)
    dis_embedding2level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix2],
                                trainable=args.trainable)(dis_in)
    dis_embedding3level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix3],
                                trainable=args.trainable)(dis_in)

    conv_result = []
    for i in range(3):
        channel_result = []
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(dis_embedding0level)
        conv1 = conv_layer(dis_embedding1level)
        conv2 = conv_layer(dis_embedding2level)
        conv3 = conv_layer(dis_embedding3level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        pooling1 = GlobalMaxPooling1D()(conv1)
        pooling2 = GlobalMaxPooling1D()(conv2)
        pooling3 = GlobalMaxPooling1D()(conv3)
        channel_result.append(pooling0)
        channel_result.append(pooling1)
        channel_result.append(pooling2)
        channel_result.append(pooling3)
        allchannel = add(channel_result)
        conv_result.append(allchannel)
    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    print("demgras_dim:{}".format(demgras_dim)) #demgras_dim:11
    print("demgras_in.shape:{}".format(demgras_in.shape))#(None, 11)
    dense_demgras = Dense(1000, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result) #numpy.concatenate函数 主要作用:沿现有的某个轴对一系列数组进行拼接。
    dense_out = Dense(1000, activation='relu', name='F1')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(500, activation='relu', name='F2')(dpt) 
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu', name='F3')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(50, activation='relu', name='F4')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(2, activation='relu', name='F5')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([dis_in,demgras_in], mode_out) #sur_in, 
    #rmsprop = optimizers.rmsprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0) 均方根传播(RMSProp) ;根据最近的权重梯度的平均值(例如变化的速度)来调整;这意味着该算法在线上和非平稳问题上表现良好(如:噪声)。
    Adam=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mae']) #rmsprop
    print(model.summary())
    return model

def train_single_channel_merge_model(model, modelpath, X_train, demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print('Training the merge model....')
    model.fit([X_train, demographics_train], y_train, epochs=200, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1)

def train_single_channel_split_model(model, modelpath, X1_train, demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print('Training the merge model....') 
    model.fit([X1_train,  demographics_train], y_train, epochs=200, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1)

def train_multi_channel_split_model(model, modelpath, X1_train,  demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print('Training the merge model....') 
    model.fit([X1_train,  demographics_train], y_train, epochs=200, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1) 

def test_single_channel_merge_model(model, modelpath,X_test,demographics_test,y_test, index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    print(r2)
    name = 'SG_merge%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

def test_single_channel_split_model(model, modelpath, X1_test,  demographics_test, y_test,index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test,  demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    name = 'SG_split%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

def test_multi_channel_split_model(model, modelpath, X1_test,  demographics_test, y_test,index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test, demographics_test], batch_size=args.batchsize, verbose=1)
    print("y_pred:{}".format(y_pred.shape))
    r2,rmse = evaluation(y_test, y_pred)
    name = 'MG_split%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)


# In[ ]:


r2_SG_merge=[]
seed = 1234

def test_single_channel_merge_model(model, modelpath,X_test,demographics_test,y_test, index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    print(r2)
    r2_SG_merge.append(r2)
    name = 'SG_merge%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

r2_SG_split=[]
def test_single_channel_split_model(model, modelpath, X1_test,  demographics_test, y_test,index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test,  demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    r2_SG_split.append(r2)
    name = 'SG_merge%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

r2_MG_merge=[]
def test_multi_channel_split_model(model, modelpath, X1_test,  demographics_test, y_test,index):
    print('Testing model...') 
    model.load_weights("/content/gdrive/MyDrive/cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test, demographics_test], batch_size=args.batchsize, verbose=1)
    print("y_pred:{}".format(y_pred.shape))
    r2,rmse = evaluation(y_test, y_pred)
    r2_MG_merge.append(r2)
    name = 'MG_split%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

for i in range(1):
    modelpath = 'SG_merge' + path
    model = single_channel_merge_model(demgras_dim)
    if not args.test:
        train_single_channel_merge_model(model, modelpath, X_train, demographics_train, y_train)
    print('testing on the testing datasets.....')
    print("modelpath:",modelpath)
    test_single_channel_merge_model(model, modelpath, X_test, demographics_test, y_test, 3)
    print('testing on the filtered testing datasets.....')
    print(r2_SG_merge)


for i in range(1):
    modelpath = 'SG_split' + path
    model = single_channel_split_model(demgras_dim)
    if not args.test:
        train_single_channel_split_model(model, modelpath, X1_train, demographics_train, y_train)
    print(f,'testing on the testing datasets.....')
    test_single_channel_split_model(model, modelpath, X1_test,  demographics_test,y_test,3)
    print('testing on the filtered testing datasets.....')
    print(r2_SG_split)


for i in range(1):
    modelpath = 'MG_merge' + path
    model= multi_channel_split_model(demgras_dim)
    if not args.test:
        train_multi_channel_split_model(model, modelpath, X1_train, demographics_train, y_train)
    extract_patientvec(model, modelpath, disease, demographics)
    print(f,'testing on the testing datasets.....')
    test_multi_channel_split_model(model, modelpath, X1_test, demographics_test,y_test,3)
    print(r2_MG_merge)


# # 对比模型2:传统机器学习

# In[ ]:


#===============================第一部分 独热编码 ============================================
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import ensemble
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
import xgboost as xgb
def onehot_RF(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    #one_hot_encoder
    print('RandomForestRegressor with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_SVR(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('SVR with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = svm.SVR()
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...') 

def onehot_AdaBoost(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('AdaBoost with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = ensemble.AdaBoostRegressor(n_estimators=10,random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_GradientBoosting(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('GradientBoosting with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = ensemble.GradientBoostingRegressor(n_estimators=10,random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_Bagging(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('Bagging with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = BaggingRegressor(n_estimators=10,random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_ExtraTree(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('ExtraTree with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = ExtraTreeRegressor(random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_DecisionTree(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('DecisionTree with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = tree.DecisionTreeRegressor(random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_KNeighbors(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('KNeighbors with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = neighbors.KNeighborsRegressor()
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_XGBoost(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    print('XGBoost with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = xgb.XGBRegressor(n_estimators=10,random_state=90)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...')

def onehot_LR(X_train, X_test, y_train,y_test, demographics_train, demographics_test):
    #one_hot_encoder
    print('LR with one_hot encoding...') 
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    #rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
    #rf.fit(train_mat, y_train.ravel())
    rf = LinearRegression(normalize=True, n_jobs=-1)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)
    print('testing on filter samples...') 

#============================== 第二部分 预测函数 单尺度、多尺度======================================================



def embedding_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #word2vec encoder
    print(f,'RandomForestRegressor with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info

    rf = ensemble.RandomForestRegressor(n_estimators=10,random_state=90)
    rf.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    
    #multichannel word2vec embedding encoder
    print(f,'RandomForestRegressor with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    rf = ensemble.RandomForestRegressor(n_estimators=10,random_state=90) #n_jobs=-1便是使用全部的CPU
    rf.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    index = 1
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)
    print('testing on filter testing samples...')

    index = 2
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)
    print('testing on filter samples...')

    index = 3
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)


def embedding_LR(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #word2vec encoder
    print(f,'Linear regression with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    print('LR_train_mat')
    print(train_mat.shape)
    lr = LinearRegression(normalize=True)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

    #multichannel word2vec embedding encoder
    print(f,'Linear regression with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = svm.SVR()
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    

def embedding_SVR(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'SVR with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    print('SVR_train_mat')
    lr = svm.SVR()
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'SVR with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = svm.SVR()
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

def embedding_AdaBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'AdaBoost with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ensemble.AdaBoostRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'AdaBoost with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ensemble.AdaBoostRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

def embedding_GradientBoosting(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'GradientBoosting with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ensemble.GradientBoostingRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'GradientBoosting with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ensemble.GradientBoostingRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)


def embedding_Bagging(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'Bagging with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = BaggingRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'Bagging with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = BaggingRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)


def embedding_ExtraTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'ExtraTree with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ExtraTreeRegressor(random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'ExtraTree with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = ExtraTreeRegressor(random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)


def embedding_DecisionTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'DecisionTree with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = tree.DecisionTreeRegressor(random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'DecisionTree with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = tree.DecisionTreeRegressor(random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)


def embedding_KNeighbors(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'KNeighbors with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = neighbors.KNeighborsRegressor()
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'KNeighbors with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = neighbors.KNeighborsRegressor()
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

def embedding_XGBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test):
    #1、word2vec encoder
    print(f,'XGBoost with embedding encoding...')
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = xgb.XGBRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    #2、multichannel word2vec embedding encoder
    print(f,'XGBoost with multichannel embedding encoding...')
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info
    lr = xgb.XGBRegressor(n_estimators=10,random_state=90)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)


# In[ ]:


onehot_SVR(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_AdaBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_GradientBoosting(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_Bagging(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_ExtraTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_DecisionTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_LR(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_KNeighbors(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
onehot_XGBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test)

embedding_SVR(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_AdaBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_GradientBoosting(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_Bagging(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_ExtraTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_DecisionTree(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_LR(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_KNeighbors(X_train, X_test,y_train,y_test,demographics_train,demographics_test)
embedding_XGBoost(X_train, X_test,y_train,y_test,demographics_train,demographics_test)

