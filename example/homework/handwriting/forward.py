# coding:utf-8
import tensorflow as tf

INPUT_NODE = 784  # image is 28*28 size pixels
OUTPUT_NODE = 10  # output index of 0~9 digitals
LAYER1_NODE = 500  # invisible layer node number


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
    if regularizer:
        tf.compat.v1.add_to_collection(
            'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
