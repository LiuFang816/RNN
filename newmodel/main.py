# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
import tensorflow as tf
from src import codereader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '../data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '../data/res/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("decode", False,
                  "Set to True for interactive decoding.")
flags.DEFINE_integer("max_length", 600,"")
flags.DEFINE_bool("generate", False, "Set to True for interactive generating.")
FLAGS = flags.FLAGS

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
    batch_size = 1
    vocab_size = 10000


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
    vocab_size = 10000


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

def read_data(input_ids_path,max_length=None):
    pass


def create_model(sess):
    pass


def main(_):
    config = get_config()

    eval_config = get_config()
    eval_config.batch_size = 1

    word_to_id = codereader.get_word_to_id(FLAGS.data_path)
    raw_data = codereader.raw_data(FLAGS.data_path, word_to_id)
    train_data, test_data, voc_size, end_id, _, _, _ = raw_data

    # 使用动态vocab_size
    config.vocab_size=voc_size
    eval_config.voc_size=voc_size

    id_to_word = codereader.reverseDic(word_to_id)

    with tf.Session() as sess:
        # Create model.
        model = create_model(sess)
        # TODO 下面这两部可以整合到rawData
        train_set = read_data(r"trainPath", FLAGS.max_length)
        test_set = read_data(r"testPath", FLAGS.max_length)

        while True:
            # 最重要的next_batch()
            true_input_ids, target_output_ids, target_weights = model.get_batch(train_set)

            _, step_loss, _ = model.step(sess,true_input_ids, target_output_ids,target_weights)





if __name__ == "__main__":
    tf.app.run()