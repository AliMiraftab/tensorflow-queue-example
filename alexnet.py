import numpy as np 
import tensorflow as tf

class Alexnet:

    def __init__(self, lr=1e-4, batch_size=512, num_batches=97):
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.g_step = tf.contrib.framework.get_or_create_global_step()

    def build(self, images):
        net = self.conv_layer(images, 11, 3, 96, strides=4, name='conv1')
        net = self.lrn_layer(net, name='lrn1')
        net = self.max_pool(net, name='pool1')

        net = self.conv_layer(net, 5, 96, 256, 'conv2')
        net = self.lrn_layer(net, name='lrn2')
        net = self.max_pool(net, name='pool2')
        
        net = self.conv_layer(net, 3, 256, 384, 'conv3')
        net = self.conv_layer(net, 3, 384, 384, 'conv4')
        net = self.conv_layer(net, 3, 384, 256, 'conv5')
        net = self.max_pool(net, name='pool5')

        dim = np.prod(net.shape[1:]).value
        net = tf.nn.relu(self.fc_layer(tf.reshape(net, [-1, dim]), dim, 4096, 'fc6'))
        net = tf.nn.relu(self.fc_layer(net, 4096, 4096, 'fc7'))
        self.logits = self.fc_layer(net, 4096, 10, 'fc8')

    def loss(self, labels):
        self.loss_op = tf.losses.sparse_softmax_cross_entropy(labels, self.logits)

    def train(self):
        self.lr = tf.train.exponential_decay(self.lr, self.g_step,
                self.batch_size * self.num_batches, 0.9, staircase=True)
        return self.optimizer(self.lr).minimize(self.loss_op, global_step=self.g_step)

    def optimizer(self, *args):
        return tf.train.GradientDescentOptimizer(*args)

    def max_pool(self, bottom, name, k=3, s=2):
        return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)
    def lrn_layer(self, bottom, name, bias=1.0, alpha=1e-4, beta=0.75):
        return tf.nn.local_response_normalization(bottom, depth_radius=2, bias=bias, alpha=alpha, beta=beta, name=name)
    def conv_layer(self, bottom, f_size, in_c, out_c, name, strides=1):
        with tf.variable_scope(name):
            f, b = self.get_conv_var(f_size, in_c, out_c, name)
            conv = tf.nn.conv2d(bottom, f, [1, strides, strides, 1], padding='SAME')
            return tf.nn.relu(tf.nn.bias_add(conv, b))
    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            w, b = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            return tf.nn.xw_plus_b(x, w, b)
    def get_conv_var(self, f_size, in_c, out_c, name):
        f = tf.get_variable(name+'_f', [f_size, f_size, in_c, out_c], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name+'_b', [out_c], initializer=tf.truncated_normal_initializer())
        return f, b
    def get_fc_var(self, in_size, out_size, name):
        w = tf.get_variable(name+'_w', [in_size, out_size], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name+'_b', [out_size], initializer=tf.truncated_normal_initializer())
        return w, b
