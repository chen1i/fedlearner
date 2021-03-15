# coding:utf-8
import os
import tensorflow as tf
import forward

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
    x = tf.compat.v1.placeholder(tf.float32, [None, forward.INPUT_NODE])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.compat.v1.get_collection('losses'))

    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExpotentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()

    with tf.compat.v1.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is %g." % (
                    step, loss_value))
                saver.save(sess, os.path_join(MODEL_SAVE_PATH,
                                              MODEL_NAME), global_step=global_step)


def main():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    backward(mnist)


if __name__ == '__main__':
    main()
