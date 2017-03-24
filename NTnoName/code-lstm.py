# -*- coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import sys

import NTnoName.codereader as codereader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", 'data/code/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'data/code/res/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("decode", False,
                  "Set to True for interactive decoding.")
FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, word_to_id=None, name=None,isDecode=False,):

        #todo 为了测试加的
        if isDecode:
            self.num_steps=len(data)
            X=[[0]]
            X[0]=data
            data = tf.convert_to_tensor(X, name="data", dtype=tf.int32)

            # data=tf.convert_to_tensor(data, name="data", dtype=tf.int32)
            self.input_data=data
            self.targets=data #这个没有用
            self.batch_size=1

            self.epoch_size=1
        else:
            self.batch_size = batch_size = config.batch_size
            self.num_steps = num_steps = config.num_steps
            #todo 替换下面的epoch_size
            # self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
            # self.input_data, self.targets,= reader.ptb_producer(
            #     data, batch_size, num_steps, name=name)

            #todo data必须为split data
            # self.X,self.Y,self.epoch_size=codereader.data_producer(data, batch_size, num_steps, 10)
            self.X,self.Y,self.epoch_size=codereader.Data_producer(data, batch_size, num_steps, word_to_id)
            # self.q = queue.Queue(maxsize=self.epoch_size)
            # for i in range(self.epoch_size):
            #     self.q.put(i)

            self.input_data, self.targets= codereader.Batch_producer(self.X, self.Y, batch_size,num_steps,self.epoch_size)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_,vocab_size):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        # vocab_size = config.vocab_size
        vocab_size=vocab_size
        self.global_step = tf.Variable(0, trainable=False)

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        #todo LSTM cell size: 隐藏层大小
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        #定义多层lstm
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            # 这里取数据(input_.input_data)，将id 转换为embedding
            #todo 如果是测试的话，input_data的格式就为 id
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        self._input_data=input_.input_data

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state

        with tf.variable_scope("RNN"):
            #todo unrolling
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1,outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

        #输出 vocab_size
        logits = tf.matmul(output, softmax_w) + softmax_b

        self._logits=logits

        # self.saver = tf.train.Saver(tf.all_variables())

        #todo test暂时注释
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        self._targets=input_.targets


        # if not is_training:
        #     return

        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            ###注意这里的train_op，run epoch里用到
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)

            #todo 新添加的暂时
            # self.saver = tf.train.Saver(tf.all_variables())


    #todo 用于测试
    def step(self,session):
        output=session.run(self._logits)
        return output


    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

#todo 这里_input_data是id
def create_decode_model(session, is_training,config,_input_data):
    """Create  model and initialize or load parameters in session."""
    decode_input = PTBInput(config=None, data=_input_data, name="DecodeInput",isDecode=True)
    model=PTBModel(is_training,config,decode_input)

    #todo test时候在这里取参数
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 10
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    # vocab_size = 500


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 500


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 500


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, name, eval_op=None, verbose=False,id_to_word=None,isDecode=False):
    """Runs the model on the given data."""

    if isDecode:
        output=model.step(session)
        return output

    start_time = time.time()
    costs = 0.0
    iters = 0
    num_steps=model.input.num_steps

    #todo 为了计算准确率
    SUM = 0
    correct_tok=0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "input_data":model._input_data,
        "targets":model._targets,
        "pred_output":model._logits
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op


    for step in range(model.input.epoch_size):
        # for step in range(1):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        #todo ☆☆ 保存checkpoint
        # if verbose:
        #     print("Saving model to %s." % FLAGS.save_path)
        #     checkpoint_path = os.path.join(FLAGS.save_path, "code.ckpt")
        #     model.saver.save(session, checkpoint_path, global_step=model.global_step)


        #todo ☆☆ add acc
        midTargets=vals["targets"]
        pred_output=vals["pred_output"]
        try:
            for i in range(model.input.batch_size):
                for j in range(num_steps):
                    SUM+=1
                    trueOutput=id_to_word[midTargets[i][j]]
                    tmp = list(pred_output[i * num_steps + j])

                    #todo top5注释
                    predOutput=id_to_word[tmp.index(max(tmp))]
                    if trueOutput==predOutput:
                        correct_tok+=1

                        # predOutput=[]
                        # for m in range(5):
                        #     index=tmp.index(max(tmp))
                        #     predOutput.append(id_to_word[index])
                        #     tmp[index]=-100
                        # if trueOutput in predOutput:
                        #     correct_tok+=1
        except:
            pass


        # if verbose and step % (model.input.epoch_size // 10) == 10:
        if step % (model.input.epoch_size // 10) == 10:
            print("-------------%s------------ "%name)

            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

            # if eval_op is None:
            # if True:
            midInputData=vals["input_data"]
            midTargets=vals["targets"]
            pred_output=vals["pred_output"]

            # print("Saving model to %s." % FLAGS.save_path)
            # sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

            try:
                for i in range(2):
                    inputStr=''
                    trueOutput=''
                    predOutput=''
                    for j in range(num_steps):
                        inputStr+=id_to_word[midInputData[i][j]]+' '

                        trueOutput+=id_to_word[midTargets[i][j]]+' '

                        tmp = list(pred_output[i * num_steps + j])
                        # tmp = list(pred_output[j * model.input.batch_size + i])
                        predOutput+=id_to_word[tmp.index(max(tmp))]+' '
                    print('Input: %s \n True Output: %s \n Pred Output: %s \n'%(inputStr,trueOutput,predOutput))
            except:
                pass


    acc=correct_tok*1.0/SUM
    print("\n%s Accuracy : %.3f"%(name,acc))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def reverseDic(curDic):
    newmaplist={}
    for key,value in curDic.items():
        newmaplist[value]=key
    return newmaplist


#todo 定义一个用于测试的函数，测试时batch_size=1 targets为NULL 关键是input 之前是PTB_Input,现在是终端读取
# def decode(word_to_id):
#
#     # init_op = tf.initialize_all_variables()
#     # sv = tf.train.Supervisor(logdir=FLAGS.save_path,init_op=init_op) #logdir用来保存checkpoint和summary
#     # saver=sv.saver #创建saver
#     with tf.Session() as sess:
#     # sv = tf.train.Supervisor(logdir=FLAGS.save_path)
#     # with sv.managed_session() as sess:
#         config=get_config()
#         # sys.stdout.write("> ")
#         # sys.stdout.flush()
#         # token = sys.stdin.read()
#         token=[]
#         token.append('for')
#         token.append('i')
#         print(token)
#         _input_data= []
#         _input_data.append(word_to_id[token[0]])
#         _input_data.append(word_to_id[token[1]])
#
#         model = create_decode_model(sess,False,config,_input_data)
#
#         output = model.step(sess)
#         print(output)



def train():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    #train_data,valid_data,test_data,word_to_id,vocabulary_size,end_id

    config = get_config()
    global num_steps
    num_steps=config.num_steps

    eval_config = get_config()

    eval_config.batch_size = 10
    eval_config.num_steps = 10
    word_to_id=codereader.get_word_to_id(FLAGS.data_path)
    raw_data =codereader.raw_data(FLAGS.data_path,word_to_id)
    # train_data, valid_data, test_data, word_to_id, _,end_id = raw_data
    train_data, test_data, word_to_id,vocab_size,end_id = raw_data
    id_to_word=reverseDic(word_to_id)

    # decode(word_to_id)

    #add split
    train_data=codereader.split_data(train_data,end_id,config.num_steps)
    test_data=codereader.split_data(test_data,end_id,config.num_steps)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data,word_to_id=word_to_id, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input,vocab_size=vocab_size)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)


        # with tf.name_scope("Test"):
        #     test_input = PTBInput(config=eval_config, data=test_data,word_to_id=word_to_id, name="TestInput")
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mtest = PTBModel(is_training=False, config=eval_config,
        #                          input_=test_input,vocab_size=vocab_size)

        #
        # with tf.name_scope("Decode"):
        #     # tf.get_variable_scope().reuse_variables()
        #     # sys.stdout.write("> ")
        #     # sys.stdout.flush()
        #     # token = sys.stdin.read()
        #     token=[]
        #     token.append('for')
        #     token.append('i')
        #     print(token)
        #     _input_data= []
        #     _input_data.append(word_to_id[token[0]])
        #     _input_data.append(word_to_id[token[1]])
        #     decode_input=PTBInput(config=None, data=_input_data, name="DecodeInput",isDecode=True)
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         decode_model=PTBModel(False,config,decode_input,vocab_size=vocab_size)
        #     # decode_model=PTBModel(False,config,decode_input)
        #
        #     # decode_model=PTBModel(False,config,decode_input)
        #     #todo test时候在这里取参数
        #     ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
        #     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #         print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #         # model.saver.restore(session, ckpt.model_checkpoint_path)
        #     else:
        #         print("Created model with fresh parameters.")
        #         # session.run(tf.global_variables_initializer())

        Config = tf.ConfigProto()
        Config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session(config=Config) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                #todo decode model Test
                # output=run_epoch(session,decode_model,'decode',id_to_word,isDecode=True)
                # # output = decode_model.step(session)
                # # print(output)
                # tmp = list(output[-1])
                # # tmp = list(pred_output[j * model.input.batch_size + i])
                # output=id_to_word[tmp.index(max(tmp))]
                # print(output)
                #------------

                train_perplexity = run_epoch(session, m,'train', eval_op=m.train_op,
                                             verbose=True,id_to_word=id_to_word)
                # if FLAGS.save_path:
                #     print("Saving model to %s." % FLAGS.save_path)
                #     sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

                #-----------todo decode model------------
                # output=run_epoch(session,decode_model,'decode',id_to_word,isDecode=True)
                # # output = decode_model.step(session)
                # tmp = list(output[-1])
                # # tmp = list(pred_output[j * model.input.batch_size + i])
                # output=id_to_word[tmp.index(max(tmp))]
                # print(output)

                #-----------------------------------------

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                test_perplexity = run_epoch(session, mtest,'test',id_to_word=id_to_word)
                print("Test Perplexity: %.3f" % test_perplexity)

            test_perplexity = run_epoch(session, mtest,'test',id_to_word=id_to_word)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

def decode():
    config = get_config()
    word_to_id=codereader.get_word_to_id(FLAGS.data_path,config.vocab_size-1)
    id_to_word=reverseDic(word_to_id)
    while True:
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
            with tf.name_scope("Decode"):
                # tf.get_variable_scope().reuse_variables()
                sys.stdout.write("> ")
                sys.stdout.flush()
                token = sys.stdin.readline().strip('\n').split(' ')
                # print(token)
                for i in range(len(token)):
                    token[i]=word_to_id[token[i]]
                decode_input=PTBInput(config=None, data=token, name="DecodeInput",isDecode=True)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    decode_model=PTBModel(False,config,decode_input)
                # decode_model=PTBModel(False,config,decode_input)

                # decode_model=PTBModel(False,config,decode_input)
                #todo test时候在这里取参数
                ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    # model.saver.restore(session, ckpt.model_checkpoint_path)
                else:
                    print("Created model with fresh parameters.")
                    # session.run(tf.global_variables_initializer())


            sv = tf.train.Supervisor(logdir=FLAGS.save_path)
            with sv.managed_session() as session:

                #todo decode model
                output=run_epoch(session,decode_model,'decode',id_to_word,isDecode=True)
                # output = decode_model.step(session)
                # print(output)
                tmp = list(output[-1])
                # tmp = list(pred_output[j * model.input.batch_size + i])
                output=id_to_word[tmp.index(max(tmp))]
                print('next token --> %s'%output)
                #------------


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
