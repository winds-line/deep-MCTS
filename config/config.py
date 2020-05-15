
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('action_num', 11, 'the number of action')
tf.flags.DEFINE_integer('train_interval', 1000, 'the train interval')
tf.flags.DEFINE_integer('max_depth', 8, 'the max depth of search tree')

tf.flags.DEFINE_float('gmma', 0.9, 'the discount rate')
tf.flags.DEFINE_float('tau', 1.0, 'the temperature parameter')

tf.flags.DEFINE_integer('num_mcts_sim', 1000, 'the max depth of search tree')
tf.flags.DEFINE_float('cpuct', 0.1, 'the max search depth of MCTS')


# for mtl network
tf.flags.DEFINE_integer('input_height', 40, 'the height of input image')
tf.flags.DEFINE_integer('input_width', 80, 'the width of input image')
tf.flags.DEFINE_integer('input_channel', 1, 'the channel of input image')
# for vsp network
tf.flags.DEFINE_float('vsp_lr', 0.00001, 'the learning rate of vsp network')
tf.flags.DEFINE_integer('vsp_batch_size', 20, 'the batch size of vsp network')
# for avp network
tf.flags.DEFINE_float('avp_lr', 0.0001, 'the learning rate of avp network')
tf.flags.DEFINE_integer('avp_batch_size', 20, 'the batch size of avp network')
