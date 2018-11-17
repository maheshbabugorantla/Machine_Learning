import tensorflow as tf
import numpy as np


def get_weights_and_bias(n_features, n_labels):
    return [tf.Variable(tf.truncated_normal((n_features, n_labels)), name='Weights'), tf.Variable(tf.zeros(n_labels), name='Bias')]


def tf_placeholder():

    x = tf.placeholder(dtype=tf.int32)
    y = tf.add(x, x)

    with tf.Session() as sess:
        print(sess.run(y, feed_dict={x: 32.0}))


def main():
    weights, bias = get_weights_and_bias(n_features=120, n_labels=5)

    with tf.Session() as sess:
        print("Weights: {}".format(sess.run(weights)))
        print("Bias: {}".format(sess.run(bias)))


if __name__ == "__main__":
    main()
