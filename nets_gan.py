# experiment: unpaired GAN with L2 loss on the low resolution images and
# only young people images in the discriminator

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
import pdb

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

# takes list of filenames and returns a 4D batch of images
# [N x W x H x C]
# also resize if necessary
def get_images(filenames, imsize=None):

    if imsize:
        batch_orig = [imresize(imread(path), (imsize, imsize), interp='bicubic') for path in filenames]
    else:
        batch_orig = [imread(path)for path in filenames]

    batch_orig_normed = np.array(batch_orig).astype(np.float32)/127.5-1

    batch_inputs = [imresize(im, 0.25, interp='bicubic') for im in batch_orig]
    # imresize returns in [0-255] so we have to normalize again
    batch_inputs_normed = np.array(batch_inputs).astype(np.float32)/127.5-1

    return batch_orig_normed, batch_inputs_normed

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def create_error_metrics(gen, inputs, origs):
    # Losses

    # metric: L2 between downsampled generated output and input
    gen_LR = slim.avg_pool2d(gen, [4, 4], stride=4, padding='SAME')
    gen_mse_LR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_LR - inputs)), 1)
    gen_L2_LR = tf.reduce_mean(gen_mse_LR)

    # metric: L2 between generated output and the original image
    gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen - origs)), 1)
    # average for the batch
    gen_L2_HR = tf.reduce_mean(gen_mse_HR)

    # metric: PSNR between generated output and original input
    gen_rmse_HR = tf.sqrt(gen_mse_HR)
    gen_PSNR = tf.reduce_mean(20*tf.log(1.0/gen_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))

    err_im_HR = gen - origs
    err_im_LR = gen_LR - inputs

    return gen_L2_LR, gen_L2_HR, gen_PSNR, err_im_LR, err_im_HR


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True, b_reuse=False):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                # huge hack
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                    self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                    with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)

        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

