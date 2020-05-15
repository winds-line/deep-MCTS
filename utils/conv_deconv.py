import tensorflow as tf
import numpy as np
from math import ceil
import math

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # with tf.device('/gpu:0'):
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope + "_bn",
                                                        reuse=True))