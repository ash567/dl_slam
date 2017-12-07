import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
import hyperparams as hyp

def encoder_decoder(inputs, pred_dim, name, nLayers, std=1e-4, do_decode=True, is_train=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        feat_stack = []
        pred_stack = []
        shape = inputs.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])
        
        # i removed this from the arg_scope. hope it doesn't break testing
        #'reuse':reuse
        
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding="VALID",
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training':is_train,
                                               'decay':0.97, # 997
                                               'epsilon':1e-5,
                                               'scale':True,
                                               'updates_collections':None},
                            stride=1,                                     
                            weights_initializer=tf.truncated_normal_initializer(stddev=std),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            # ENCODER
            net = inputs
            chans = 32
            # first, one conv at full res
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
            net = slim.conv2d(net, chans, [3, 3], stride=1,
                              scope='conv%d' % 0)
            feat_stack.append(net)
            print_shape(net)
            for i in range(nLayers):
                chans = int(chans*2)
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
                net = slim.conv2d(net, chans, [3, 3], stride=2,
                                  scope='conv%d_1' % (i+1))
                print_shape(net)
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
                net = slim.conv2d(net, chans, [3, 3], stride=1,
                                  scope='conv%d_2' % (i+1))
                print_shape(net)
                if i != nLayers-1:
                    feat_stack.append(net)
                h = int(h/2)
                w = int(w/2)
            if do_decode:
                # DECODER
                for i in reversed(range(nLayers)):
                    # predict from these feats
                    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
                    pred = slim.conv2d(net, pred_dim, [3, 3], stride=1,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       scope='pred%d' % (i+1))
                    pred_stack.append(pred)
                    print_shape(pred)
                    # deconv the feats
                    chans = int(chans/2)

                    # undo that pad
                    net = tf.slice(net,[0,1,1,0],[-1,h,w,-1])
                    print_shape(net)

                    h = int(h*2)
                    w = int(w*2)
                    if hyp.do_classic_deconv:
                        net = slim.conv2d_transpose(net, chans, [4, 4], stride=2, padding="SAME", scope="deconv%d" % (i+1))
                        print_shape(net)
                    else:
                        net = tf.image.resize_nearest_neighbor(net, [h, w],
                                                               name="upsamp%d" % (i+1))
                        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
                        net = slim.conv2d(net, chans, [3, 3], stride=1,
                                          scope='conv%d' % (i+1))
                    print_shape(net)
                    print_shape(net)
                    # concat [upsampled pred, deconv, saved conv from earlier]
                    net = tf.concat(3,[tf.image.resize_images(pred, [h, w]),
                                       net,
                                       feat_stack.pop()],name="concat%d" % (i+1))
                    print_shape(net)
            # one last pred at full res
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
            pred = slim.conv2d(net, pred_dim, [3, 3], stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='pred%d' % 0)
            pred_stack.append(pred)
            print_shape(pred)
        return pred_stack
