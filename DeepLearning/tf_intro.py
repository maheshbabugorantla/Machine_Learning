import tensorflow as tf


def main():

    hello_constant = tf.constant('Hello World')

    with tf.Session() as sess:
        output = sess.run(hello_constant)
        print(output)


if __name__ == '__main__':
    main()
