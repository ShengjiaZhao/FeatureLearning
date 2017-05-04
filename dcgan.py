import tensorflow as tf
import numpy as np
import time
import subprocess
import math
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = tf.contrib.layers.convolution2d(x, 64, [4, 4], [2, 2],
                                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                activation_fn=tf.identity)
        conv1 = lrelu(conv1)
        conv2 = tf.contrib.layers.convolution2d(conv1, 128, [4, 4], [2, 2],
                                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                activation_fn=tf.identity)
        conv2 = tf.contrib.layers.batch_norm(conv2)
        conv2 = lrelu(conv2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = tf.contrib.layers.fully_connected(conv2, 1024,
                                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                activation_fn=tf.identity)
        fc1 = tf.contrib.layers.batch_norm(fc1)
        fc1 = lrelu(fc1)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return tf.sigmoid(fc2), fc2


def generator(z):
    with tf.variable_scope('g_net') as vs:
        fc1 = tf.contrib.layers.fully_connected(z, 1024,
                                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                activation_fn=tf.identity)
        fc1 = tf.contrib.layers.batch_norm(fc1)
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.contrib.layers.fully_connected(fc1, 7 * 7 * 128,
                                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                activation_fn=tf.identity)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        fc2 = tf.contrib.layers.batch_norm(fc2)
        fc2 = tf.nn.relu(fc2)
        conv1 = tf.contrib.layers.convolution2d_transpose(fc2, 64, [4, 4], 2,
                                                          weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                          activation_fn=tf.identity)
        conv1 = tf.contrib.layers.batch_norm(conv1)
        conv1 = tf.nn.relu(conv1)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, [4, 4], 2, activation_fn=tf.sigmoid)
        # mask = np.ones((28, 28, 1))

        # output = tf.concat([output[:, 3:, :, :], tf.ones(tf.stack([tf.shape(output)[0], 3, 28, 1]))], 1)
        # output[:, 10:13, 10:13, :] = 0.0
        return output


class GenerativeAdversarialNet(object):
    def __init__(self):
        self.hidden_num = 100
        self.z = tf.placeholder(tf.float32, [None, self.hidden_num])
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.g = generator(self.z)
        self.d, self.d_logits = discriminator(self.x)
        self.d_, self.d_logits_ = discriminator(self.g, reuse=True)

        self.d_loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits, labels=tf.ones_like(self.d)))
        self.d_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_, labels=tf.zeros_like(self.d_)))
        self.d_loss = self.d_loss_x + self.d_loss_g
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_, labels=tf.ones_like(self.d_)))
        self.loss = self.d_loss + self.g_loss

        self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
        self.d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        # self.d_gradient_ = tf.gradients(self.d_loss_g, self.g)[0]
        # self.d_gradient = tf.gradients(self.d_logits, self.x)[0]

        self.merged = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss_x', self.d_loss_x),
            tf.summary.scalar('d_loss_g', self.d_loss_g),
            tf.summary.scalar('loss', self.loss)
        ])

        # self.image = tf.summary.image('generated images', self.g, max_images=10)
        self.saver = tf.train.Saver(tf.global_variables())

        self.fig, self.ax = None, None

    def visualize(self, batch_size, sess):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.cla()
        bz = np.random.normal(-1, 1, [batch_size, self.hidden_num]).astype(np.float32)
        image = sess.run(self.g, feed_dict={self.z: bz})
        num_row = int(math.floor(math.sqrt(batch_size)))
        canvas = np.zeros((28*num_row, 28*num_row))
        for i in range(num_row):
            for j in range(num_row):
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = image[i*num_row+j, :, :, 0]
        self.ax.imshow(canvas, cmap=plt.get_cmap('Greys'))
        plt.draw()
        plt.pause(0.01)

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            if os.path.isdir('log/train'):
                subprocess.call('rm -rf log/train'.split())
            os.makedirs('log/train')
            summary_writer = tf.summary.FileWriter('./train')
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                self.visualize(batch_size, sess)
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    bx, _ = mnist.train.next_batch(batch_size)
                    bx = np.reshape(bx, [batch_size, 28, 28, 1])
                    bz = np.random.normal(-1, 1, [batch_size, self.hidden_num]).astype(np.float32)
                    d_loss, _ = sess.run([self.d_loss, self.d_train], feed_dict={self.x: bx, self.z: bz})
                    d_loss_g, d_loss_x = sess.run([self.d_loss_g, self.d_loss_x], feed_dict={self.x: bx, self.z: bz})
                    g_loss, _ = sess.run([self.g_loss, self.g_train], feed_dict={self.z: bz})
                    g_loss, _ = sess.run([self.g_loss, self.g_train], feed_dict={self.z: bz})
                    g_loss, = sess.run([self.g_loss], feed_dict={self.z: bz})

                    merged = sess.run(self.merged, feed_dict={self.x: bx, self.z: bz})
                    summary_writer.add_summary(merged, epoch * batch_idxs + idx)

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_x: %.8f, d_loss_g: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss_x, d_loss_g, g_loss))

                    if idx == 500:
                        self.visualize(batch_size, sess)
                if os.path.isdir('log/model'):
                    subprocess.call('rm -rf log/model'.split())
                os.makedirs('log/model')
                self.saver.save(sess, 'log/model', global_step=epoch)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    c = GenerativeAdversarialNet()
    c.train()
