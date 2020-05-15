# The VSP network predicts the future state of the vehicle based on a given control action
import numpy as np
from config.config import *
from utils.netutil import *


class AvpNet:
    def __init__(self, sess):
        self.lr = FLAGS.avp_lr
        self.batch_size = FLAGS.avp_batch_size
        self.sess = sess

        self.action_num = FLAGS.action_num
        self._build_net()

    # build the network architecture
    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder('float', shape=[None, FLAGS.input_height, FLAGS.input_width, 1])
        self.p_label = tf.placeholder('float', shape=[None, 11])
        self.v_label = tf.placeholder('float', shape=[None, 1])
        c_names, w_initializer, b_initializer = \
            ['avp_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
            tf.truncated_normal_initializer(mean=0.0, stddev=0.01), \
            tf.constant_initializer(0)  # config of layers
        with tf.variable_scope('avp_net'):
            with tf.variable_scope('l1'):
                w_c1 = tf.get_variable('w_c1', [8, 8, 1, 32], initializer=w_initializer, collections=c_names)
                b_c1 = tf.get_variable('b_c1', [32], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(conv2d(self.s, w_c1, 4) + b_c1)
                h_pool1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('l2'):
                w_c2 = tf.get_variable('w_c2', [4, 4, 32, 64], initializer=w_initializer, collections=c_names)
                b_c2 = tf.get_variable('b_c2', [64], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(conv2d(h_pool1, w_c2, 2) + b_c2)
                h_pool2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('l3'):
                w_c3 = tf.get_variable('w_c3', [3, 3, 64, 64], initializer=w_initializer, collections=c_names)
                b_c3 = tf.get_variable('b_c3', [64], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(conv2d(h_pool2, w_c3, 1) + b_c3)

                h_pool3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                h_conv3_out_size = np.prod(h_pool3.get_shape().as_list()[1:])
                h_conv3_flat = tf.reshape(h_pool3, [-1, h_conv3_out_size])

            with tf.variable_scope('l4'):
                w_f1 = tf.get_variable('w_f1', [h_conv3_out_size, 128], initializer=w_initializer,
                                       collections=c_names)
                b_f1 = tf.get_variable('b_f1', [128], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(h_conv3_flat, w_f1) + b_f1)

            with tf.variable_scope('l5'):
                w_a_f2_p = tf.get_variable('w_a_f2_p', [128, self.action_num], initializer=w_initializer,
                                           collections=c_names)
                b_a_f2_p = tf.get_variable('b_a_f2_p', [self.action_num], initializer=b_initializer, collections=c_names)
                self.p = tf.nn.softmax(tf.matmul(l4, w_a_f2_p) + b_a_f2_p)

                w_a_f2_v = tf.get_variable('w_a_f2_v', [128, 1], initializer=w_initializer,
                                           collections=c_names)
                b_a_f2_v = tf.get_variable('b_a_f2_v', [1], initializer=b_initializer, collections=c_names)
                self.v = tf.matmul(l4, w_a_f2_v) + b_a_f2_v

        with tf.variable_scope('loss'):
            self.loss_v = tf.reduce_mean(tf.squared_difference(self.v, self.v_label))
            self.loss_p = tf.reduce_mean(tf.reduce_sum(-self.p_label*tf.log(self.p), axis=1))
            self.loss = self.loss_p + self.loss_v
        with tf.variable_scope('train'):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='avp_net')
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            self._train_op = optimizer.minimize(self.loss, var_list=train_vars)

    def predict(self, obs):
        obs = np.array(obs)
        obs = np.reshape(obs, (1, FLAGS.input_height, FLAGS.input_width, 1))
        p, v = self.sess.run([self.p, self.v], feed_dict={self.s: obs})
        return p[0, :], v[0][0]

    def learn(self, p_batch, v_batch, obs_batch):
        v_batch = np.array(v_batch)
        v_batch = v_batch[:, np.newaxis]
        _, loss = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: obs_batch,
                                           self.p_label: p_batch,
                                           self.v_label: v_batch})
        return loss

