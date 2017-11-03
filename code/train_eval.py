
# coding: utf-8

# In[1]:

import tensorflow as tf

from config import cfg
from utils import get_batch_data
from capsLayer import primary_caps, digit_caps
from utils import load_mnist
from utils import save_images
import numpy as np


class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.Y_label = get_batch_data()
                self.Y = tf.one_hot(self.Y_label, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()


                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                learning_rate = tf.train.exponential_decay(cfg.learning_rate, self.global_step,
                                                       cfg.step_size, cfg.learning_rate_decay,
                                                       staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)

                # set up adam optimizer with default setting
                self._optimizer = tf.train.AdamOptimizer(learning_rate)
                gradidents = self._optimizer.compute_gradients(self.total_loss)

                self.train_op = self._optimizer.apply_gradients(gradidents,
                                                             global_step=self.global_step)
                
                
                ############# ACC ###############
                self.match = tf.equal(self.Y_label, self.argmax_idx)
                self.correct_num = tf.reduce_sum(tf.to_float(self.match))
                
                self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
                self.summary_test_acc = tf.summary.scalar('test/acc', self.test_acc)

            else:
                self.X = tf.placeholder(tf.float32,
                                        shape=(cfg.batch_size, 28, 28, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            self.caps1 = primary_caps(conv1)

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            self.caps2 = digit_caps(self.caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2)+1e-7)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert argmax_idx.get_shape() == [cfg.batch_size, ]

            # c). indexing
            # It's not easy to understand the indexing process with argmax_idx
            # as we are 3-dim animal
            masked_v = []
            argmax_idx = tf.reshape(argmax_idx, shape=(cfg.batch_size, ))
            self.argmax_idx = argmax_idx
            
            for batch_size in range(cfg.batch_size):
                v = self.caps2[batch_size][argmax_idx[batch_size], :]
                masked_v.append(tf.reshape(v, shape=(1, 16)))

            self.masked_v = tf.concat(masked_v, axis=0)
            assert self.masked_v.get_shape() == [cfg.batch_size, 16]

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

        
    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        self.max_l = tf.reduce_mean(max_l)
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        self.max_r = tf.reduce_mean(max_r)
        self.v_length_sum = tf.reduce_sum(self.v_length)
        assert max_l.get_shape() == [cfg.batch_size, 10]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        #max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        #max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(tf.reduce_sum(squared, axis=-1))

        # 3. Total loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # Summary
        summary_train = []
        #tf.summary.scalar('b_IJ', tf.reduce_sum(tf.abs(self.b_IJ)))
        summary_train.append(tf.summary.scalar('margin_loss', self.margin_loss))
        summary_train.append(tf.summary.scalar('reconstruction_loss', self.reconstruction_err))
        summary_train.append(tf.summary.scalar('total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        summary_train.append(tf.summary.image('reconstruction_img', recon_img))
        self.summary_train = tf.summary.merge(summary_train)


# In[2]:


capsNet = CapsNet(is_training=cfg.is_training)
tf.logging.info('Graph loaded')
sv = tf.train.Supervisor(graph=capsNet.graph,
                         logdir=cfg.logdir,
                         save_model_secs=0, save_summaries_secs=30, summary_op=None)
teX, teY = load_mnist(cfg.dataset, False)
with sv.managed_session() as sess:
    num_batch = int(60000 / cfg.batch_size)
    num_test_batch = int(10000 / cfg.batch_size)
    for epoch in range(cfg.epoch):
        if sv.should_stop():
            break
        for step in range(num_batch):
            global_step = sess.run(capsNet.global_step)
                
            if step % cfg.summary_step == 0:
                _, summary_train = sess.run([capsNet.train_op, capsNet.summary_train])
                sv.summary_writer.add_summary(summary_train, global_step=global_step)
            else:
                sess.run(capsNet.train_op)
            
            if step % cfg.test_step == 0:
                cor_all = 0
                for step_test in xrange(num_test_batch):
                    start = step_test * cfg.batch_size
                    end = start + cfg.batch_size
                    cor_num = sess.run(capsNet.correct_num, feed_dict={capsNet.X:teX[start:end], capsNet.Y_label:teY[start:end]})
                    cor_all += cor_num
                test_acc = cor_all/10000
                print "Test Acc on %f epoch %d %d/%d "%(test_acc, epoch, step, num_batch)
                summary_test_acc = sess.run(capsNet.summary_test_acc, feed_dict={capsNet.test_acc:test_acc})
                sv.summary_writer.add_summary(summary_test_acc, global_step=global_step)
        sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

tf.logging.info('Training done')


# In[3]:




# if __name__ == '__main__':
#     capsNet = CapsNet(is_training=cfg.is_training)
#     tf.logging.info('Graph loaded')

#     teX, teY = load_mnist(cfg.dataset, cfg.is_training)
#     with capsNet.graph.as_default():
#         sv = tf.train.Supervisor(logdir=cfg.logdir)
#         # with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         with sv.managed_session() as sess:
#             sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
#             tf.logging.info('Restored')

#             reconstruction_err = []
#             for i in range(10000 // cfg.batch_size):
#                 start = i * cfg.batch_size
#                 end = start + cfg.batch_size
#                 recon_imgs = sess.run(capsNet.decoded, {capsNet.X: teX[start:end]})
#                 orgin_imgs = np.reshape(teX[start:end], (cfg.batch_size, -1))
#                 squared = np.square(recon_imgs - orgin_imgs)
#                 reconstruction_err.append(np.mean(squared))

#                 if i % 5 == 0:
#                     imgs = np.reshape(recon_imgs, (cfg.batch_size, 28, 28, 1))
#                     size = 6
#                     save_images(imgs[0:size * size, :], [size, size], 'results/test_%03d.png' % i)
#             print('test acc:')
#             print((1. - np.mean(reconstruction_err)) * 100)

