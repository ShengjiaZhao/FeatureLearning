import tensorflow as tf
import numpy as np
import time
import subprocess
import math
from matplotlib import pyplot as plt
from abstract_network import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def conv_pc_layer(x, y_real=None, channel=256, reuse=False, name='layer', activation='relu'):
    with tf.variable_scope(name) as vs:
        if reuse:
            vs.reuse_variables()
        if activation == 'relu':
            y = conv2d_t_bn_relu(x, channel, 4, 2)
        else:
            y = tf.nn.sigmoid(conv2d_t_bn(x, channel, 4, 2))

        if y_real is not None:
            with tf.variable_scope('grad'):
                # Gradient w.r.t. y
                y_grad_x = conv2d_t_bn_lrelu(x, channel, 4, 2)
                y_grad_y = conv2d_bn_lrelu(y - y_real, channel, 4, 1)
                y_grad = tf.concat([y_grad_x, y_grad_y], axis=-1)
                y_grad = conv2d(y_grad, channel, 4, 1)

                # Gradient w.r.t. x
                x_grad_y = conv2d_bn_lrelu(y - y_real, channel, 4, 2)
                x_grad_x = conv2d_bn_lrelu(x, channel, 4, 1)
                x_grad = tf.concat([x_grad_y, x_grad_x], axis=-1)
                x_grad = conv2d(x_grad, x.get_shape()[-1].value, 4, 1)
                return y, x_grad, y_grad
        else:
            return y


def fc_pc_layer(x, y_real=None, dim=256, reuse=False, name='layer'):
    with tf.variable_scope(name) as vs:
        if reuse:
            vs.reuse_variables()
        y = fc_bn_relu(x, dim)

        if y_real is not None:
            with tf.variable_scope('grad'):
                # Gradient w.r.t. y
                y_grad_x = fc_bn_lrelu(x, dim)
                y_grad_y = fc_bn_lrelu(y - y_real, dim)
                y_grad = tf.concat([y_grad_x, y_grad_y], axis=-1)
                y_grad = fc_layer(y_grad, dim)

                # Gradient w.r.t. x
                x_grad_y = fc_bn_lrelu(y - y_real, dim)
                x_grad_x = fc_bn_lrelu(x, dim)
                x_grad = tf.concat([x_grad_y, x_grad_x], axis=-1)
                x_grad = fc_layer(x_grad, x.get_shape()[-1].value)
                return y, x_grad, y_grad
        else:
            return y


def network_step(z_3, z_2, z_1, x, summary_step=None, reuse=False):
    z_2_pred, grad_3_2, grad_2_3 = fc_pc_layer(z_3, z_2, 7*7*128, name='g3', reuse=reuse)
    z_3_new = z_3 + grad_3_2
    z_2_new = z_2 + grad_2_3
    loss3 = tf.reduce_mean(tf.abs(z_2_pred - z_2))
    if summary_step is not None:
        tf.summary.scalar('loss3@step%d' % summary_step, loss3)

    z_2_conv = tf.reshape(z_2, [-1, 7, 7, 128])
    z_1_pred, grad_2_1, grad_1_2 = conv_pc_layer(z_2_conv, z_1, 64, name='g2', reuse=reuse)
    z_2_new += tf.reshape(grad_2_1, [-1, 7*7*128])
    z_1_new = z_1 + grad_1_2
    loss2 = tf.reduce_mean(tf.abs(z_1_pred - z_1))
    if summary_step is not None:
        tf.summary.scalar('loss2@step%d' % summary_step, loss2)

    z_0_pred, grad_1_0, grad_0_1 = conv_pc_layer(z_1, x, 1, activation='sigmoid', name='g1', reuse=reuse)
    z_1_new += grad_1_0
    loss1 = 10.0 * tf.reduce_mean(tf.abs(z_0_pred - x))
    if summary_step is not None:
        tf.summary.scalar('loss1@step%d' % summary_step, loss1)
    return z_3_new, z_2_new, z_1_new, loss1 + loss2 + loss3

def generative_net(z_3):
    z_2 = fc_pc_layer(z_3, dim=7*7*128, name='g3', reuse=True)
    z_2_conv = tf.reshape(z_2, [-1, 7, 7, 128])
    z_1 = conv_pc_layer(z_2_conv, channel=64, name='g2', reuse=True)
    return conv_pc_layer(z_1, channel=1, activation='sigmoid', name='g1', reuse=True)

def get_updates(x, batch_size=64):
    z_3 = tf.random_normal([batch_size, 128])
    z_2 = tf.random_normal([batch_size, 7*7*128])
    z_1 = tf.random_normal([batch_size, 14, 14, 64])
    z_3_list, z_2_list, z_1_list = [], [], []
    steps = 5
    total_loss = 0.0
    for i in range(steps):
        z_3, z_2, z_1, step_loss = network_step(z_3, z_2, z_1, x, summary_step=i, reuse=(i != 0))
        z_3_list.append(z_3)
        z_2_list.append(z_2)
        z_1_list.append(z_1)
        if i != 0:
            total_loss = 0.5 * total_loss + step_loss
            sparse_loss = tf.reduce_mean(tf.abs(z_3)) + tf.reduce_mean(tf.abs(z_2)) + tf.reduce_mean(tf.abs(z_1))
            total_loss += 0.2 * sparse_loss
    return z_3_list, z_2_list, z_1_list, total_loss

def display_conv_weights(weights, name):
    weights = tf.transpose(weights, perm=[3, 0, 1, 2])
    channels, height, width = [weights.get_shape()[i].value for i in range(3)]
    print(channels, height, width)
    cnt = int(math.floor(math.sqrt(channels)))
    print(weights.get_shape())
    weights = tf.slice(weights, [0, 0, 0, 0], [cnt*cnt, -1, -1, 1])
    print(weights.get_shape())
    weights = tf.pad(weights, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    weights = tf.transpose(weights, perm=[1, 0, 2, 3])

    weights = tf.reshape(weights, [height+2, cnt, cnt, width+2])
    weights = tf.transpose(weights, perm=[1, 0, 2, 3])
    weights = tf.reshape(weights, [1, (height+2)*cnt, (width+2)*cnt, 1])
    tf.summary.image(name, weights, max_outputs=1)

class PredictiveCoder:
    def __init__(self, name='pcoder_sparse'):
        self.name = name
        self.logdir = 'log/%s' % self.name

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.z_3_list, self.z_2_list, self.z_1_list, self.loss = get_updates(self.x)
        self.z_3 = tf.placeholder(tf.float32, [None, 128])
        self.gen = generative_net(self.z_3)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
            self.loss, var_list=[v for v in tf.global_variables() if 'grad' not in v.name])
        self.train_grad_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
            self.loss, var_list=[v for v in tf.global_variables() if 'grad' in v.name])
        tf.summary.scalar('loss', self.loss)

        # Fetch convolution weights and transform them for display
        conv1_weights = [v for v in tf.global_variables() if v.name == "g1/Conv2d_transpose/weights:0"][0]
        display_conv_weights(conv1_weights, 'conv1_weights')
        conv2_weights = [v for v in tf.global_variables() if v.name == "g2/Conv2d_transpose/weights:0"][0]
        display_conv_weights(conv2_weights, 'conv2_weights')

        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables())
        self.fig, self.ax = None, None

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            if os.path.isdir(self.logdir):
                subprocess.call(('rm -rf %s' % self.logdir).split())
            os.makedirs(self.logdir)
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            summary_writer.flush()
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    if idx % 200 == 0:
                        self.visualize(batch_size, sess)

                    bx, _ = mnist.train.next_batch(batch_size)
                    bx = np.reshape(bx, [batch_size, 28, 28, 1])
                    loss, _, _ = sess.run([self.loss, self.train_op, self.train_grad_op], feed_dict={self.x: bx})

                    if idx % 10 == 0:
                        merged = sess.run(self.merged, feed_dict={self.x: bx})
                        summary_writer.add_summary(merged, epoch * batch_idxs + idx)

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, loss))

                modeldir = '%s/models' % self.logdir
                if os.path.isdir(modeldir):
                    subprocess.call(('rm -rf %s' % modeldir).split())
                os.makedirs(modeldir)
                self.saver.save(sess, modeldir, global_step=epoch)

    def visualize(self, batch_size, sess):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.cla()
        bz = np.random.normal(-1, 1, [batch_size, 128]).astype(np.float32)
        image = sess.run(self.gen, feed_dict={self.z_3: bz})
        num_row = int(math.floor(math.sqrt(batch_size)))
        canvas = np.zeros((28*num_row, 28*num_row))
        for i in range(num_row):
            for j in range(num_row):
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = image[i*num_row+j, :, :, 0]
        self.ax.imshow(canvas, cmap=plt.get_cmap('Greys'))
        plt.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    c = PredictiveCoder()
    c.train()
