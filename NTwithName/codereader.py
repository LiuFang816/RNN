# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import collections

#读取文件，将token以list形式保存
def _read_words(filename):
    with tf.gfile.GFile(filename,'r') as f:
        return f.read().decode('utf-8').replace("\r\n"," ENDMARKER ").split(' ')
        # return f.read().decode('utf-8').split(' ')
#
#创建词汇表，将token按出现频率排序，转换为word:id
#todo 必须要考虑变量名的问题  否则UNK占比太高
def _build_vocab(filename,n=None):
    data = _read_words(filename)
    # print(data)
    counter = collections.Counter(data)
    # print(counter[''])
    # print(sum(counter.values()))
    # a=sum(counter.values())
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))
    # print(len(values))
    # print(sum(values[0:9999]))
    # b=sum(values[0:9999])
    # print(b*1.0/a)

    #取频率较高的n个word
    words=words[0:4999]
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['UNK']=len(words)
    return word_to_id

#将文件中的token转换为id
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    wordId=[]
    for word in data:
        if word in word_to_id:
            wordId.append(word_to_id[word])
        else:
            wordId.append(word_to_id['UNK'])
            # print("NOT IN VOC")
    # return [word_to_id[word] for word in data if word in word_to_id]
    return wordId

#获取转换为id的数据（train,valid,test)
def raw_data(data_path,word_to_id):
    train_path=os.path.join(data_path,"new_train.txt")
    # valid_path=os.path.join(data_path,"validGT10.txt")
    test_path=os.path.join(data_path,"new_test.txt")

    word_to_id=word_to_id
    train_data=_file_to_word_ids(train_path,word_to_id)
    # valid_data=_file_to_word_ids(valid_path,word_to_id)
    test_data=_file_to_word_ids(test_path,word_to_id)

    vocabulary_size=len(word_to_id)
    end_id=word_to_id['ENDMARKER']
    return train_data,test_data,word_to_id,vocabulary_size,end_id

def get_word_to_id(data_path=None):
    train_path=os.path.join(data_path,"train.txt")
    word_to_id=_build_vocab(train_path)
    return word_to_id

#将数据转换为n*numstep的形式，line[n+1]是line[n]左移一位（每个代码段末尾一行除外）
def split_data(train_data,ENDMARKER_id,num_step):
    _split_train_data=[]
    index=0
    for i in range(len(train_data)):
        if train_data[i]==ENDMARKER_id:
            _split_train_data.append(train_data[index:i+1])
            index=i+1
    if index!=len(train_data) and len(train_data)-index>=num_step:
        _split_train_data.append(train_data[index:len(train_data)])

    new_split_data=[]
    for i in range(len(_split_train_data)):
        if len(_split_train_data[i])>num_step:
            for index in range(len(_split_train_data[i])-num_step+1):
                new_split_data.append(_split_train_data[i][index:index+num_step])
        else:
            new_split_data.append(_split_train_data[i])
    return new_split_data


#todo 消去UNK多的batch 后面解决变量名问题之后需要进行调整
def Data_producer(data,batch_size,numsteps,word_to_id):
    # print(len(data))
    # print(word_to_id['ENDMARKER'])
    x = []
    y = []
    for i in range(len(data)):
        # print(i)
        # countUNK=0
        # for j in range(numsteps):
        #     if data[i][j]==word_to_id['UNK']:
        #         countUNK+=1
        # # 设置比例
        # if countUNK*1.0/numsteps>=0.5:
        #     continue
        if data[i][numsteps-1]==word_to_id['ENDMARKER']:
            continue
        else:
            x.append(data[i])
            y.append(data[i+1])
    epoch_size=len(x)//batch_size
    return x,y,epoch_size



def Batch_producer(X,Y,batchsize,num_steps,epoch_size):
    X = tf.convert_to_tensor(X, name="X", dtype=tf.int32)
    Y = tf.convert_to_tensor(Y, name="Y", dtype=tf.int32)
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(X, [i*batchsize,0], [batchsize, num_steps])
    y = tf.slice(Y, [i*batchsize,0], [batchsize, num_steps])
    return x,y

#TEST
# X,Y,epochsize=Data_producer(_split_train_data,20,10,word_to_id)
# # x,y=producer(X,Y,0,10)
# print(X)
# print(Y)
# print(epochsize)
# q = queue.Queue(maxsize=epochsize)
# for i in range(epochsize):
#     q.put(i)
# for i in range(5):
#     print(batch_producer(X,Y,q.get()))


f=open('D:/py_project/Tensorflow/myEx/RNN/NTwithName/data/train.txt')
f1=open('D:/py_project/Tensorflow/myEx/RNN/NTwithName/data/new_train.txt','w')
lineNUM=0
while 1:
    line=f.readline()
    if not line:
        break
    code=line.split(' ')
    if lineNUM<600:
        if len(code)<=600 and len(code)>=11:
            f1.write(line)
            lineNUM+=1
    else: break




