# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import collections


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)

        ##############新写法
        # batch_len表示一个batch有多少行数据
        # 比如共有2000行数据 numstep=60 batch_size=20 那么batch_len=100
        # 将data reshape成[batch_len * num_steps] 表示100*60个字 也就是前100行的内容
        batch_len = data_len // num_steps // batch_size
        data = tf.reshape(raw_data[0:batch_len * num_steps * batch_size],
                          [batch_size,num_steps * batch_len])
        epoch_size=(num_steps * batch_len - 1) // num_steps

        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps-1])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps-1])
        return x, y

def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode('utf-8').replace("\r\n", " ENDMARKER ").split(' ')


def _file_to_word_ids(filename, word_to_id, max_length=None):
    word_ids = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            words = line.split(" ")
            if max_length and len(words) >= max_length:
                continue
            for word in words:
                if word in word_to_id:
                    word_ids.append(word_to_id[word])
                else:
                    word_ids.append(word_to_id['UNK'])
            word_ids.append(word_to_id['ENDMARKER'])
            if max_length:
                for i in range(max_length - len(words)-1):
                    word_ids.append(word_to_id['PAD'])
    return word_ids


def raw_data(data_path=None, word_to_id=None, max_length=None):
    train_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "test.txt")

    word_to_id = word_to_id
    train_data= _file_to_word_ids(train_path, word_to_id, max_length)
    test_data= _file_to_word_ids(test_path, word_to_id, max_length)

    vocabulary_size = len(word_to_id)
    end_id = word_to_id['ENDMARKER']
    left_id = word_to_id['{']
    right_id = word_to_id['}']
    END_ID = word_to_id['ENDTOK']
    PAD_ID = word_to_id['PAD']
    return train_data, test_data,vocabulary_size, end_id, END_ID, left_id, right_id, PAD_ID


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))
    words = words[0:4997]
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['ENDTOK'] = len(word_to_id)
    word_to_id['UNK'] = len(word_to_id)
    word_to_id['PAD'] = len(word_to_id)
    return word_to_id


def get_word_to_id(data_path=None):
    train_path = os.path.join(data_path, "train.txt")
    word_to_id = _build_vocab(train_path)
    return word_to_id


def reverseDic(curDic):
    newmaplist = {}
    for key, value in curDic.items():
        newmaplist[value] = key
    return newmaplist


def weight_producer(weights):
    q = tf.FIFOQueue(500000, 'float')
    init = q.enqueue_many((weights,))
    w = q.dequeue()
    return init, w
