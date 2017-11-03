import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from config import cfg


def primary_caps(inputs, kernel_size=9, stride=2, vec_len=8, output_num=32):
    pri = slim.conv2d(inputs, output_num*vec_len, kernel_size, stride, padding='VALID', activation_fn=None)
    pri = tf.reshape(pri, [inputs.shape[0].value, -1, vec_len])
    pri = squash(pri)
    return pri


def digit_caps(inputs, vec_len=16, output_num=10):
    W_ij = tf.get_variable('W_ij', shape=[inputs.shape[-2].value, output_num, inputs.shape[-1].value, vec_len], dtype=tf.float32)
    fn_init = tf.zeros([inputs.shape[-2].value, output_num, 1, vec_len])

    inputs = tf.reshape(inputs, [inputs.shape[0].value, inputs.shape[1].value, 1, 1, inputs.shape[2].value])
    predicts = tf.scan(lambda ac, x: tf.matmul(x, W_ij), # [32*6*6, 10, 1, 8] [32*6*6, 10, 8, 16] -> [32*6*6, 10, 1, 16]
                             tf.tile(inputs, [1, 1, output_num, 1, 1]), #[b, 32*6*6, 10, 1, 8]
                             initializer=fn_init, name='predicts')
    predicts = tf.squeeze(predicts, axis=3) #[b, 32*6*6, 10, 16]

    v_IJ, b_IJ = routing(predicts)
    return v_IJ

def routing(inputs):

    b_IJ = tf.constant(np.zeros([inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, 1], dtype=np.float32))

    assert inputs.get_shape() == [cfg.batch_size, 6*6*32, 10, 16]

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [1, 1, 1152, 10, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            assert c_IJ.get_shape() == [cfg.batch_size, 6*6*32, 10, 1]

            # line 5:
            # weighting u_hat with c_IJ, element-wise in the last tow dim
            # => [batch_size, 1152, 10, 16, 1]
            s_J = tf.multiply(c_IJ, inputs)
            assert s_J.get_shape() == [cfg.batch_size, 6*6*32, 10, 16]
            # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
            s_J = tf.reduce_sum(s_J, axis=1)
            assert s_J.get_shape() == [cfg.batch_size, 10, 16]

            # line 6:
            # squash using Eq.1,
            v_J = squash(s_J)
            assert v_J.get_shape() == [cfg.batch_size, 10, 16]

            # line 7:
            # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
            # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
            # batch_size dim, resulting in [1, 1152, 10, 1, 1]
            #v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            v_J_tiled = tf.expand_dims(v_J, axis = 1)
            u_produce_v = tf.reduce_sum(inputs*v_J_tiled, axis=-1, keep_dims=True)
            assert u_produce_v.get_shape() == [cfg.batch_size, 6*6*32, 10, 1]
            b_IJ += u_produce_v

    return(v_J, b_IJ)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len],
    Returns:
        A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm+1e-7)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
