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

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import new_reader
import Stack
# import newmodel.new_reader as new_reader
# import newmodel.Stack as Stack
import sys
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", 'data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'data/res/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("decode", False,
                  "Set to True for interactive decoding.")
flags.DEFINE_bool("generate", False, "Set to True for interactive generating.")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None,isDecode=False):
        if isDecode:
            self.num_steps=len(data)+1
            X=[[0]]
            X[0]=data
            data = tf.convert_to_tensor(X, name="data", dtype=tf.int32)
            self.input_data=data
            self.targets=data #这个没有用
            self.batch_size=1
            self.epoch_size=1
        else:
            self.batch_size = batch_size = config.batch_size
            self.num_steps = num_steps = config.num_steps
            self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
            self.input_data, self.targets = new_reader.ptb_producer(
                data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_, START_MARK, END_MARK, PAD_MARK):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps - 1
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
            # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            #     lstm_cell, output_keep_prob=config.keep_prob)
        # cell=lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())
        self.state_stack = Stack.Stack()

        with tf.device("/gpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        self._input_data = input_.input_data

        test = []
        self._test = test
        # self._inputs=inputs

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

        # def func_push(state,batch):
        #     self.state_stack.push(state)
        #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]
        #
        # def func_pop(batch):
        #     state = self.state_stack.pop()
        #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]
        #
        # def func_default(state,batch):
        #     return state[0][0][batch], state[0][1][batch], state[1][0][batch], state[1][1][batch]

        def func_push(state):
            self.state_stack.push(state)
            return state[0][0], state[0][1], state[1][0], state[1][1]

        def func_pop():
            state = self.state_stack.pop()
            return state[0][0], state[0][1], state[1][0], state[1][1]


        def func_default(state):
            return state[0][0], state[0][1], state[1][0], state[1][1]

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                # tf.case({tf.equal(self._input_data[0][time_step], START_MARK): lambda :func_push(state),
                #          tf.equal(self._input_data[0][time_step], END_MARK): lambda :func_pop()},
                #         default=lambda :func_default(state),
                #         exclusive=False)

                # state1=[]
                # state2=[]
                # state3=[]
                # state4=[]
                # for batch in range(batch_size):
                #     s1,s2,s3,s4=tf.cond(tf.equal(self._input_data[batch][time_step], START_MARK), lambda: func_push(state,batch),lambda: func_default(state,batch))
                #     s1,s2,s3,s4=tf.cond(tf.equal(self._input_data[batch][time_step], END_MARK), lambda: func_pop(batch),lambda :func_default(state,batch))
                #     state1.append(s1)
                #     state2.append(s2)
                #     state3.append(s3)
                #     state4.append(s4)
                # ss1=tf.reshape(state1,[batch_size,size])
                # ss2=tf.reshape(state2,[batch_size,size])
                # ss3=tf.reshape(state3,[batch_size,size])
                # ss4=tf.reshape(state4,[batch_size,size])
                # state=((ss1,ss2),(ss3,ss4))

                #todo RNN 注释
                # new_state=tf.cond(tf.equal(self._input_data[0][time_step], START_MARK), lambda: func_push(state),lambda: func_default(state))
                # new_state=tf.cond(tf.equal(self._input_data[0][time_step], END_MARK), lambda: func_pop(),lambda :func_default(state))
                # state=((new_state[0],new_state[1]),(new_state[2],new_state[3]))

                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs,1), [-1, size])
        # output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        # loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._targets = input_.targets
        self._logits = logits

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def step(self,session):
        output=session.run(self._logits)
        return output

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


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 800
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 5000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 1024
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 200
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1024
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 200
    vocab_size = 10000


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


def run_epoch(session, model, eval_op=None, verbose=False, id_to_word=None,end_id=None,isDecode=False):
    """Runs the model on the given data."""
    if isDecode:
        output=model.step(session)
        return output
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    SUM = 0
    correct_tok=0

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "input_data": model._input_data,
        "targets": model._targets,
        "pred_output": model._logits
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
    # for step in range(1):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        # todo 禁止data自动更新，而在run之前手动更新一次inputData


        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        #todo add 计算accuracy
        midInputData = vals["input_data"]
        midTargets=vals["targets"]
        pred_output=vals["pred_output"]
        try:
            for i in range(model.input.batch_size):
                for j in range(model.input.num_steps-1):
                    SUM+=1
                    trueOutput=id_to_word[midTargets[i][j]]
                    tmp = list(pred_output[i * (model.input.num_steps-1) + j])
                    #todo topN这里注释
                    predOutput=id_to_word[tmp.index(max(tmp))]
                    if midInputData[i][j]==end_id:
                        SUM-=1
                        break
                    if trueOutput==predOutput:
                        correct_tok+=1
                    # todo topN使用这里
                    # predOutput=[]
                    # for m in range(N):
                    #     index=tmp.index(max(tmp))
                    #     predOutput.append(id_to_word[index])
                    #     tmp[index]=-100
                    # if trueOutput in predOutput:
                    #     correct_tok+=1
        except:
            print("--------ERROR 计算acc----")
            pass

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

            if True:
                midInputData = vals["input_data"]
                midTargets = vals["targets"]
                pred_output = vals["pred_output"]
                global num_steps
                try:
                    for i in range(1):
                        inputStr = ''
                        trueOutput = ''
                        predOutput = ''
                        for j in range(num_steps):
                            inputStr += id_to_word[midInputData[i][j]] + ' '

                            trueOutput += id_to_word[midTargets[i][j]] + ' '

                            tmp = list(pred_output[i * num_steps + j])
                            predOutput += id_to_word[tmp.index(max(tmp))] + ' '
                        print('Input: %s \n True Output: %s \n Pred Output: %s \n' % (inputStr, trueOutput, predOutput))
                except:
                    pass

    acc=correct_tok*1.0/SUM
    print("Accuracy : %.3f"%acc)
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


num_steps = 0


def train():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    config = get_config()


    word_to_id = new_reader.get_word_to_id(FLAGS.data_path)
    # todo raw_data还应包含weights
    raw_data = new_reader.raw_data(FLAGS.data_path, word_to_id, config.num_steps)
    train_data, test_data,voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
    id_to_word = new_reader.reverseDic(word_to_id)

    config = get_config()
    global num_steps
    num_steps = config.num_steps-1

    eval_config = get_config()
    # eval_config.batch_size = 1
    # eval_config.num_steps = 1

    # 使用动态vocab_size
    # config.vocab_size=voc_size
    # eval_config.voc_size=voc_size


    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input,
                             START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # with tf.name_scope("Valid"):
        #     valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        #     tf.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input,
                                 START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

        Config = tf.ConfigProto()
        Config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session(config=Config) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True, id_to_word=id_to_word,end_id=end_id)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                # valid_perplexity = run_epoch(session, mvalid, id_to_word=id_to_word)
                # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, id_to_word=id_to_word,end_id=end_id)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

def decode():
    choice=['1','2','3','4','5','q']
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    config = get_config()

    word_to_id = new_reader.get_word_to_id(FLAGS.data_path)
    # todo raw_data还应包含weights
    raw_data = new_reader.raw_data(FLAGS.data_path, word_to_id, config.num_steps)
    train_data, test_data,voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
    id_to_word = new_reader.reverseDic(word_to_id)

    config = get_config()
    global num_steps
    num_steps = config.num_steps-1

    sys.stdout.write("> ")
    sys.stdout.flush()
    token = sys.stdin.readline().strip('\n').split(' ')
    for i in range(len(token)):
        if token[i] not in word_to_id:
            token[i]=word_to_id['UNK']
        else:
            token[i]=word_to_id[token[i]]

    while True:
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
            for i in range(len(token)):
                print(id_to_word[token[i]],end=' ')
            print('\n')

            with tf.name_scope("Train"):
                decode_input = PTBInput(config=config, data=token, name="TrainInput",isDecode=True)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    decode_model = PTBModel(is_training=True, config=config, input_=decode_input,
                                 START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)
                # tf.summary.scalar("Training Loss", m.cost)
                # tf.summary.scalar("Learning Rate", m.lr)

                ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    # model.saver.restore(session, ckpt.model_checkpoint_path)
                else:
                    print("Created model with fresh parameters.")
            sv = tf.train.Supervisor(logdir=FLAGS.save_path)
            with sv.managed_session() as session:
                output=run_epoch(session,decode_model,'decode',id_to_word,end_id=end_id,isDecode=True)
                # output = decode_model.step(session)
                # print(output)
                tmp = list(output[-1])

                # output=id_to_word[tmp.index(max(tmp))]
                # print('next token --> %s'%output)
                #------------

                #todo 输出top5
                predOutput=[]
                count=0
                while count<5:
                    index=tmp.index(max(tmp))
                    if index==PAD_MARK:
                        tmp[index]=-100
                        continue
                    predOutput.append(id_to_word[index])
                    count+=1
                    tmp[index]=-100

                print('next token --> ')
                for i in range(len(predOutput)):
                    print('%d: %s'%(i+1,predOutput[i]))
                print('You Choose: ')
                x = sys.stdin.readline().strip('\n')
                if x!='q' and x in choice:
                    token.append(word_to_id[predOutput[int(x)-1]])
                elif x not in choice:
                    if x not in word_to_id:
                        token.append(word_to_id['UNK'])
                    else:
                        token.append(word_to_id[x])
                else:
                    break

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()
if __name__ == "__main__":
    tf.app.run()
