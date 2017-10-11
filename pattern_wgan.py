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
from dataset import *

def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def discriminator(x, size=1024, layers=2, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        net = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        for layer in range(layers):
            net = fc_lrelu(net, size)
        return tf.contrib.layers.fully_connected(net, 1, activation_fn=tf.identity)


def generator(z, size=1024, layers=2):
    with tf.variable_scope('g_net') as vs:
        net = z
        for layer in range(layers):
            net = fc_bn_relu(net, size)
        output = tf.contrib.layers.fully_connected(net, 28*28, activation_fn=tf.sigmoid)
        return tf.reshape(output, [-1, 28, 28, 1])


class GenerativeAdversarialNet(object):
    def __init__(self, dataset, gsize=1024, dsize=1024, layers=2, name="gan"):
        self.dataset = dataset
        self.name = name
        self.hidden_num = 10
        self.z = tf.placeholder(tf.float32, [None, self.hidden_num])
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])

        self.g = generator(self.z, size=gsize, layers=layers)
        self.d = discriminator(self.x, size=dsize, layers=layers)
        self.d_ = discriminator(self.g, size=dsize, layers=layers, reuse=True)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.g
        d_hat = discriminator(x_hat, size=dsize, layers=layers, reuse=True)

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
        self.log_path, self.fig_path = self.make_log_path()
        self.fig_cnt = 0

    def visualize(self, batch_size, sess):
        bz = np.random.normal(-1, 1, [batch_size, self.hidden_num]).astype(np.float32)
        image = sess.run(self.g, feed_dict={self.z: bz})
        num_row = int(math.floor(math.sqrt(batch_size)))
        canvas = np.zeros((28*num_row, 28*num_row))
        for i in range(num_row):
            for j in range(num_row):
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = image[i*num_row+j, :, :, 0]
        misc.imsave("%s/%d.png" % (self.fig_path, self.fig_cnt), canvas)
        self.fig_cnt += 1

    def make_log_path(self):
        log_path = "log/%s" % self.name
        if os.path.isdir(log_path):
            subprocess.call(('rm -rf %s' % log_path).split())
        os.makedirs(log_path)
        fig_path = "%s/fig" % log_path
        os.makedirs(fig_path)
        return log_path, fig_path

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            summary_writer = tf.summary.FileWriter(self.log_path)
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    if idx % 500 == 0:
                        self.visualize(batch_size, sess)

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--netname', type=str, default='')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--gsize', type=int, default=1024)
    parser.add_argument('--dsize', type=int, default=32)

    args = parser.parse_args()

    if args.gpus is not '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.netname == '':
        args.netname = 'gan_%s_%d_%d_%d' % (args.dataset, args.layers, args.gsize, args.dsize)
    if args.dataset == 'mnist':
        dataset = MnistDataset()
    elif args.dataset == 'symmetric':
        dataset = PatternDataset(type=args.type)
    else:
        print("Unknown dataset")
        exit(-1)
    c = GenerativeAdversarialNet(dataset, gsize=args.gsize, dsize=args.dsize, layers=args.layers, name=args.netname)
    c.train()


# Use histogram between the two distributions as a numerical metric of fitting accuracy
# Even though for many patterns the log likelihood can be accurately estimated, it is a poor indicator of visual appeal
# Use a low dimensional noise vector so that it is impossible for generator to represent the full distribution
# Study the relationship of generator capacity, discriminator capacity vs. quality
# Question: 1. verify that invariant discriminator is the reason for GAN success
# 2. study the relationship between discriminator form and the invariance it encodes
