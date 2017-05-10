import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import time
import subprocess
import math, os
from matplotlib import pyplot as plt
from pattern_generator import *
import scipy.misc as misc
from abstract_network import *

def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)

def conv_discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2


def mlp_discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_lrelu(x, 1024)
        fc2 = fc_lrelu(fc1, 1024)
        fc3 = fc_lrelu(fc2, 1024)
        fc4 = tf.contrib.layers.fully_connected(fc3, 1, activation_fn=tf.identity)
        return fc4


def conv_generator(z):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_bn_relu(fc2, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


def mlp_generator(z):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, 1024)
        fc3 = fc_bn_relu(fc2, 1024)
        output = tf.contrib.layers.fully_connected(fc3, 28*28, activation_fn=tf.sigmoid)
        return tf.reshape(output, [-1, 28, 28, 1])


class GenerativeAdversarialNet(object):
    def __init__(self, dataset, name="gan"):
        self.dataset = dataset
        self.hidden_num = 10
        self.z = tf.placeholder(tf.float32, [None, self.hidden_num])
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])

        if "mlpgen" in name:
            generator = mlp_generator
        else:
            generator = conv_generator
        if "mlpdisc" in name:
            discriminator = mlp_discriminator
        else:
            discriminator = conv_discriminator

        self.g = generator(self.z)
        self.d = discriminator(self.x)
        self.d_ = discriminator(self.g, reuse=True)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.g
        d_hat = discriminator(x_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
        self.d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        self.d_loss_x = -tf.reduce_mean(self.d)
        self.d_loss_g = tf.reduce_mean(self.d_)
        self.d_loss = self.d_loss_x + self.d_loss_g + self.d_grad_loss
        self.g_loss = -tf.reduce_mean(self.d_)
        self.loss = self.d_loss + self.g_loss

        self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
        self.d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_vars)
        self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_vars)

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
        self.model_path = "log/%s" % name
        self.fig_path = "%s/fig" % self.model_path

    def visualize(self, batch_size, sess, save_idx):
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
        misc.imsave("%s/%d.png" % (self.fig_path, save_idx), canvas)
        self.ax.imshow(canvas, cmap=plt.get_cmap('Greys'))
        plt.draw()
        plt.pause(0.01)

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            if os.path.isdir(self.model_path):
                subprocess.call(('rm -rf %s' % self.model_path).split())
            os.makedirs(self.model_path)
            os.makedirs(self.fig_path)
            summary_writer = tf.summary.FileWriter(self.model_path)
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    if idx % 500 == 0:
                        self.visualize(batch_size, sess, epoch * 2 + idx / 500)

                    bx = self.dataset.next_batch(batch_size)
                    bx = np.reshape(bx, [batch_size, 28, 28, 1])
                    bz = np.random.normal(-1, 1, [batch_size, self.hidden_num]).astype(np.float32)
                    d_loss, _, d_loss_g, d_loss_x = sess.run([self.d_loss, self.d_train, self.d_loss_g, self.d_loss_x],
                                                             feed_dict={self.x: bx, self.z: bz})
                    g_loss, _ = sess.run([self.g_loss, self.g_train], feed_dict={self.z: bz})

                    merged = sess.run(self.merged, feed_dict={self.x: bx, self.z: bz})
                    summary_writer.add_summary(merged, epoch * batch_idxs + idx)

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_x: %.8f, d_loss_g: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss_x, d_loss_g, g_loss))

                save_path = "%s/model" % self.model_path
                if os.path.isdir(save_path):
                    subprocess.call(('rm -rf %s' % save_path).split())
                os.makedirs(save_path)
                self.saver.save(sess, save_path, global_step=epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--type', type=str, default='symmetric')
    parser.add_argument('--netname', type=str, default='')

    args = parser.parse_args()

    if args.gpus is not '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.netname == '':
        args.netname = 'gan_%s' % args.type
    dataset = PatternDataset(type=args.type)
    c = GenerativeAdversarialNet(dataset, name=args.netname)
    c.train()


# Use histogram between the two distributions as a numerical metric of fitting accuracy
# Even though for many patterns the log likelihood can be accurately estimated, it is a poor indicator of visual appeal
# Use a low dimensional noise vector so that it is impossible for generator to represent the full distribution
# Study the relationship of generator capacity, discriminator capacity vs. quality
# Question: 1. verify that invariant discriminator is the reason for GAN success
# 2. study the relationship between discriminator form and the invariance it encodes
