from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import dtypes
import sys
from dump2disk_gan import *
from utils import *
from nets_modified import *
from saverloader import *

# CHA
# import hyperparams_gan as hyp

import batcher as bat
from nets_gan import *
import hyperparams_gan as hyp

from deeplab_resnet import prepare_label # , ImageReader

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
# from saverloader_gan import *
# from ops import *
# if hyp.dataset_name == 'KITTI':
#     zrt2flow = zrt2flow_kitti


class Minuet(object):
    def __init__(self, sess,
                 checkpoint_dir=None,
                 vis_dir=None,
                 log_dir=None):
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.vis_dir = vis_dir
        self.log_dir = log_dir
        self.d_bn1 = batch_norm(name='d_depth_bn1')
        self.d_bn2 = batch_norm(name='d_depth_bn2')
        self.d_bn3 = batch_norm(name='d_depth_bn3')

        self.build_model()

        # variable naming convention

        # i == image
        # s == segmentation (classes)
        # d == depth
        # v == valid

        # f == flow
        # m == motion mask (maybe it isn't consistent right now, but this should be 1's at moving objects)
        
        # then...

        # _g == ground truth 
        # _e == estimate
        
        # then...
        # _z == visualization
        
        # or (just when batching)...
        # _t == training
        # _v == validation

    def build_model(self):
        with tf.variable_scope("inputs"):
            # get batches
            if hyp.dataset_name == 'VKITTI':
                (self.i1_g_t, self.i2_g_t,
                 self.s1_g_t, self.s2_g_t,
                 self.d1_g_t, self.d2_g_t,
                 self.f12_g_t, self.f23_g_t,
                 self.v1_g_t, self.v2_g_t,
                 self.p1_g_t, self.p2_g_t,
                 self.m1_g_t, self.m2_g_t,
                 self.off_h_t, self.off_w_t,
                 _, _, _, _, _,
                 _, _, _, _, _,
                 _, _, _, _)\
                 = bat.vkitti_batch(hyp.dataset_t,
                                    hyp.bs,hyp.h,hyp.w,
                                    shuffle=True)

                self.fy_t = tf.cast(tf.tile(tf.reshape(hyp.fy,[1]),[hyp.bs]),tf.float32)
                self.fx_t = tf.cast(tf.tile(tf.reshape(hyp.fx,[1]),[hyp.bs]),tf.float32)
                self.y0_t = tf.cast(tf.tile(tf.reshape(hyp.y0,[1]),[hyp.bs]),tf.float32)
                self.x0_t = tf.cast(tf.tile(tf.reshape(hyp.x0,[1]),[hyp.bs]),tf.float32)
                
                (self.i1_g_v, self.i2_g_v,
                 self.s1_g_v, self.s2_g_v,
                 self.d1_g_v, self.d2_g_v,
                 self.f12_g_v, self.f23_g_v,
                 self.v1_g_v, self.v2_g_v,
                 self.p1_g_v, self.p2_g_v,
                 self.m1_g_v, self.m2_g_v,
                 self.off_h_v, self.off_w_v,
                 _, _, _, _, _,
                 _, _, _, _, _,
                 _, _, _, _)\
                 = bat.vkitti_batch(hyp.dataset_v,
                                    hyp.bs,hyp.h,hyp.w,
                                    shuffle=True)
                self.fy_v = tf.cast(tf.tile(tf.reshape(hyp.fy,[1]),[hyp.bs]),tf.float32)
                self.fx_v = tf.cast(tf.tile(tf.reshape(hyp.fx,[1]),[hyp.bs]),tf.float32)
                self.y0_v = tf.cast(tf.tile(tf.reshape(hyp.y0,[1]),[hyp.bs]),tf.float32)
                self.x0_v = tf.cast(tf.tile(tf.reshape(hyp.x0,[1]),[hyp.bs]),tf.float32)
                
                (self.i1_g_n, self.i2_g_n,
                 self.s1_g_n, self.s2_g_n,
                 self.d1_g_n, self.d2_g_n,
                 self.f12_g_n, self.f23_g_n,
                 self.v1_g_n, self.v2_g_n,
                 self.p1_g_n, self.p2_g_n,
                 self.m1_g_n, self.m2_g_n,
                 self.off_h_n, self.off_w_n,
                 _, _, _, _, _,
                 _, _, _, _, _,
                 _, _, _, _)\
                 = bat.vkitti_batch(hyp.dataset_gan,
                                    hyp.bs,hyp.h,hyp.w,
                                    shuffle=True)

                (self.i1_g_g, self.i2_g_g,
                 self.s1_g_g, self.s2_g_g,
                 self.d1_g_g, self.d2_g_g,
                 self.f12_g_g, self.f23_g_g,
                 self.v1_g_g, self.v2_g_g,
                 self.p1_g_g, self.p2_g_g,
                 self.m1_g_g, self.m2_g_g,
                 self.off_h_g, self.off_w_g,
                 _, _, _, _, _,
                 _, _, _, _, _,
                 _, _, _, _)\
                 = bat.vkitti_batch(hyp.dataset_gan_g,
                                    hyp.bs,hyp.h,hyp.w,
                                    shuffle=True)


                # self.i1_g_n,self.i2_g_n, \
                #     self.s1_g_n,self.s2_g_n, \
                #     self.d1_g_n,self.d2_g_n, \
                #     self.f12_g_n,self.f23_g_n, \
                #     self.v1_g_n,self.v2_g_n, \
                #     self.p1_g_n,self.p2_g_n, \
                #     self.m1_g_n,self.m2_g_n, \
                #     self.off_h_n,self.off_w_n = bat.kitti_batch(hyp.dataset_gan,hyp.bs,hyp.h,hyp.w,hyp.shuffle_val)


                self.fy_n = tf.cast(tf.tile(tf.reshape(hyp.fy,[1]),[hyp.bs]),tf.float32)
                self.fx_n = tf.cast(tf.tile(tf.reshape(hyp.fx,[1]),[hyp.bs]),tf.float32)
                self.y0_n = tf.cast(tf.tile(tf.reshape(hyp.y0,[1]),[hyp.bs]),tf.float32)
                self.x0_n = tf.cast(tf.tile(tf.reshape(hyp.x0,[1]),[hyp.bs]),tf.float32)

            # elif hyp.dataset_name == 'KITTI':
            #     self.i1_g_t,self.i2_g_t, \
            #         self.p1_g_t,self.p2_g_t, \
            #         self.fy_t, self.fx_t, \
            #         self.y0_t, self.x0_t, \
            #         self.h_t, self.w_t, \
            #         self.off_h_t,self.off_w_t = bat.kitti_batch(hyp.dataset_t,hyp.bs,hyp.h,hyp.w,True)
            #     self.i1_g_v,self.i2_g_v, \
            #         self.p1_g_v,self.p2_g_v, \
            #         self.fy_v, self.fx_v, \
            #         self.y0_v, self.x0_v, \
            #         self.h_v, self.w_v, \
            #         self.off_h_v,self.off_w_v = bat.kitti_batch(hyp.dataset_v,hyp.bs,hyp.h,hyp.w,hyp.shuffle_val)
            #     # use ones for the vars we don't have
            #     self.s1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.s2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.s1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.s2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.d1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.d2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.d1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.d2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.f12_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 2))
            #     self.f23_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 2))
            #     self.f12_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 2))
            #     self.f23_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 2))
            #     self.v1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.v2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.v1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.v2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w,1))
            #     self.m1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.m2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.m1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
            #     self.m2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))

            # get placeholders. we'll fill these with either train or val data
            self.i1_g = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,3],name='i1_g')
            self.i2_g = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,3],name='i2_g')
            self.s1_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.s2_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.d1_g = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,1],name='d1_g')
            self.d2_g = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,1],name='d2_g')
            self.f12_g = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,2],name='f12_g')
            self.v1_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.v2_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.m1_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.m2_g = tf.placeholder(tf.float32,[hyp.bs, hyp.h, hyp.w, 1])
            self.p1_g = tf.placeholder(tf.float32, [hyp.bs,4,4], name='p1_g')
            self.p2_g = tf.placeholder(tf.float32, [hyp.bs,4,4], name='p2_g')
            self.off_h = tf.placeholder(tf.int32, [hyp.bs], name='off_h')
            self.off_w = tf.placeholder(tf.int32, [hyp.bs], name='off_w')
            self.fy = tf.placeholder(tf.float32, [hyp.bs], name='fy')
            self.fx = tf.placeholder(tf.float32, [hyp.bs], name='fx')
            self.y0 = tf.placeholder(tf.float32, [hyp.bs], name='y0')
           
            self.x0 = tf.placeholder(tf.float32, [hyp.bs], name='x0')
            self.d1_gan = tf.placeholder(tf.float32,[hyp.bs,hyp.h,hyp.w,1],name='d1_gan')
   
            # the provided transforms map points from the camera frame to world frame
            # we need transforms that map points from one camera to the other
            self.rt12_g = ominus(self.p2_g, self.p1_g)
            self.rt21_g = ominus(self.p1_g, self.p2_g)

            # the camera intrinsics are provided in the full-res setup. we need to fix that.
            self.fy = (self.fy*hyp.scale)
            self.fx = (self.fx*hyp.scale)
            self.y0 = (self.y0*hyp.scale)-tf.cast(self.off_h,tf.float32)
            self.x0 = (self.x0*hyp.scale)-tf.cast(self.off_w,tf.float32)
            # self.y0 = tf.ones_like(self.y0)
            # self.x0 = tf.ones_like(self.x0)
            
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        depth_stuff, flow_stuff, odo_stuff, seg_stuff = self.inference(self.i1_g,self.i2_g,
                                                            self.d1_g,self.d2_g,
                                                            self.f12_g,
                                                            self.v1_g,
                                                            self.m1_g,
                                                            self.rt12_g,
                                                            self.s1_g,
                                                            self.off_h, self.off_w,
                                                            self.fy, self.fx,
                                                            self.y0, self.x0,
                                                            is_train=hyp.do_train,
                                                            reuse=False)

        if hyp.do_depth:
            depth_loss, d1_e, _ = depth_stuff
        if hyp.do_flow:
            flow_loss, f12_e, fi1_e = flow_stuff
        if hyp.do_odo:
            odo_loss, rt_loss, self.rt12_e, o12_e, o12_g, oi1_e, oi1_g = odo_stuff
        if hyp.do_seg:
            reduced_loss_seg, prediction_seg_prob, predictions = seg_stuff

        if hyp.do_gan:

            if hyp.do_depth_gan:

                # If this flag is set, insted of feeding the depth from the depth net, it feeds the original depth
                if hyp.do_depth_gan_original:
                    gan_input_e = std_d1_e = (self.d1_g - 4.5)/5.0
                else:
                    gan_input_e = std_d1_e = (d1_e - 4.5)/5.0

                self.gan_input_g = self.std_d1_gan = (self.d1_gan  - 4.5)/5.0

            if hyp.do_seg_gan:
                # Making the range of label values from 0 to 13 and then reducing the last but 1 dimention    
                self.s1_g_n_1hot = tf.to_float(tf.squeeze(tf.one_hot(tf.to_int32(self.s1_g_n - 1), hyp.num_seg_classes), [3]))
                self.s1_g_1hot = tf.to_float(tf.squeeze(tf.one_hot(tf.to_int32(self.s1_g - 1), hyp.num_seg_classes), [3]))

                if hyp.do_seg_gan_original:
                    prediction_seg_prob_feed = self.s1_g_1hot
                else:
                    prediction_seg_prob_feed = self.seg_e_prob

                if hyp.do_depth_gan:
                    self.gan_input_g = tf.concat(concat_dim = 3, values = [self.s1_g_n_1hot, self.std_d1_gan], name = 'concat_gan_g')
                    gan_input_e = tf.concat(concat_dim = 3, values = [prediction_seg_prob_feed, std_d1_e], name = 'concat_gan_e')
                else:
                    self.gan_input_g = self.s1_g_n_1hot
                    gan_input_e = prediction_seg_prob_feed

            gan_output_e = self.discriminator(gan_input_e)
            self.gan_output_g = self.discriminator(self.gan_input_g, reuse=True)

            self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.gan_output_g, tf.ones_like(self.gan_output_g)))
            self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(gan_output_e, tf.zeros_like(gan_output_e)))
            self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(gan_output_e, tf.ones_like(gan_output_e)))
            self.d_loss = (self.d_loss_real + self.d_loss_fake)/2.0
        else:
            self.g_loss = self.d_loss = 0

        ## start backward stuff
        if hyp.do_backward and (hyp.do_flow or hyp.do_odo):
            # for some of these, going backward doesn't make a differencen
            _, flow_stuff2, odo_stuff2, _ = self.inference(self.i2_g,self.i1_g,
                                                        self.d2_g,self.d1_g,
                                                        self.f12_g, # f21_g doesn't exist
                                                        self.v2_g,
                                                        self.m2_g,
                                                        self.rt21_g,
                                                        self.s2_g,
                                                        self.off_h, self.off_w,
                                                        self.fy, self.fx,
                                                        self.y0, self.x0,
                                                        is_train=hyp.do_train,
                                                        reuse=True)
        else:
            flow_stuff2 = flow_stuff
            odo_stuff2 = odo_stuff
        if hyp.do_flow:
            flow_loss2, f21_e, fi2_e = flow_stuff2
        if hyp.do_odo:
            odo_loss2, rt_loss2, self.rt21_e, o21_e, o21_g, oi2_e, oi2_g = odo_stuff2
        # end backward stuff

        with tf.variable_scope("losses"):
            self.unsup_loss = 0.0
            self.loss = 0.0

            if hyp.do_depth:
                self.loss += hyp.depth_coeff*depth_loss
            if hyp.do_seg:
                self.loss += hyp.seg_coeff * reduced_loss_seg

            if hyp.do_flow:
                flow_l2_loss, flow_photo_loss, flow_smooth_loss = flow_loss
                _, flow_photo_loss2, flow_smooth_loss2 = flow_loss2 # f2f1 l2 is wrong
                flow_photo_loss = (flow_photo_loss+flow_photo_loss2)/2
                flow_smooth_loss = (flow_smooth_loss+flow_smooth_loss2)/2
                self.loss += hyp.flow_l2_coeff*flow_l2_loss + \
                             hyp.flow_photo_coeff*flow_photo_loss + \
                             hyp.flow_smooth_coeff*flow_smooth_loss
                self.unsup_loss += hyp.flow_photo_coeff*flow_photo_loss + \
                                   hyp.flow_smooth_coeff*flow_smooth_loss

            if hyp.do_odo:
                rtd, rta, td, ta = rt_loss
                rtd2, rta2, td2, ta2 = rt_loss2
                rtd = (rtd+rtd2)/2
                rta = (rta+rta2)/2
                td = (td+td2)/2
                ta = (ta+ta2)/2
                odo_photo_loss = (odo_loss+odo_loss2)/2
                # odo_photo_loss = (odo_photo_loss+odo_photo_loss2)/2
                self.loss += hyp.odo_photo_coeff*odo_photo_loss + \
                             hyp.rtd_coeff*rtd + \
                             hyp.rta_coeff*rta + \
                             hyp.td_coeff*td + \
                             hyp.ta_coeff*ta
                self.unsup_loss += hyp.odo_photo_coeff * odo_photo_loss
                self.rtd = rtd
                self.rta = rta
                self.td = td
                self.ta = ta

        with tf.name_scope("visualizations"):
            self.i1_g_z = tf.cast((self.i1_g+0.5)*255,tf.uint8)
            self.i2_g_z = tf.cast((self.i2_g+0.5)*255,tf.uint8)
            self.m1_g_z = oned2color(self.m1_g)

            if hyp.do_flow:
                self.f12_e_z = flow2color(f12_e)
                self.f12_g_z = flow2color(self.f12_g)
                self.fi1_e_z = tf.cast((fi1_e+0.5)*255,tf.uint8)
                if hyp.do_backward:
                    self.f21_e_z = flow2color(f21_e)
                    self.fi2_e_z = tf.cast((fi2_e+0.5)*255,tf.uint8)
            
            if hyp.do_depth:
                self.d1_e_z = oned2color(d1_e)
                self.d1_g_z = oned2color(self.d1_g)

            if hyp.do_odo:
                self.o12_e_z = flow2color(o12_e)
                self.o12_g_z = flow2color(o12_g)
                self.oi1_e_z = tf.cast((oi1_e+0.5)*255,tf.uint8)
                if hyp.do_backward:
                    self.o21_e_z = flow2color(o21_e)
                    self.o21_g_z = flow2color(o21_g)
                    self.oi2_e_z = tf.cast((oi2_e+0.5)*255,tf.uint8)

            if hyp.do_seg:
                self.seg_e_prob = prediction_seg_prob
                self.seg_e_pred = predictions
                self.seg_loss = reduced_loss_seg

        with tf.variable_scope("summaries"):
            loss_sum = tf.summary.scalar("loss", self.loss)
            if hyp.do_depth:
                with tf.name_scope("depth_summ"):

                    d1_e_sum = tf.summary.histogram("d1_e", tf.exp(d1_e))
                    d1_g_sum = tf.summary.histogram("d1_g", tf.exp(self.d1_g))
                    d_vis_summary_d = tf.summary.image('images/depth',
                                     tf.concat(concat_dim=1, values=[oned2color(self.d1_g), oned2color(d1_e)]))
                    d_vis_summary_i = tf.summary.image('images/rgb', self.i1_g)
                    d_vis_summary_i1plus2 = tf.summary.image('images/depth_average', (self.d1_g + self.d2_g)/2.0)
                    depth_loss_sum = tf.summary.scalar("depth_loss", depth_loss)
                    scaled_depth_loss_sum = tf.summary.scalar("scaled_depth_loss",
                                                              hyp.depth_coeff*depth_loss)
            if hyp.do_seg:
                with tf.name_scope("seg_summ"):
                    seg_loss_sum = tf.summary.scalar("seg_loss", self.seg_loss)
                    # labels_summary = decode_labels(self.s1_g, label_colours)
                    # preds_summary = decode_labels(self.seg_e_pred, label_colours)
                    # vis_summary = tf.summary.image('images', 
                                                     # tf.concat(concat_dim=2, values=[labels_summary, preds_summary]), 
                                                     # max_outputs=hyp.save_num_images) # Concatenate row-wise.    

            if hyp.do_gan:
                with tf.name_scope("gan_summ"):
                    gan_gen_loss_sum = tf.summary.scalar("gan_generator_loss", self.g_loss)
                    gan_dis_loss_sum = tf.summary.scalar("gan_discriminator_loss", self.d_loss)

            if hyp.do_flow:
                with tf.name_scope("flow_summ"):
                    f12_e_sum = tf.summary.histogram("f12_e", f12_e)
                    f12_g_sum = tf.summary.histogram("f12_g", self.f12_g)
                    flow_l2_loss_sum = tf.summary.scalar("flow_l2_loss", flow_l2_loss)
                    flow_photo_loss_sum = tf.summary.scalar("flow_photo_loss", flow_photo_loss)
                    flow_smooth_loss_sum = tf.summary.scalar("flow_smooth_loss", flow_smooth_loss)
                    scaled_flow_l2_loss_sum = tf.summary.scalar("scaled_flow_l2_loss",
                                                                hyp.flow_l2_coeff*flow_l2_loss)
                    scaled_flow_photo_loss_sum = tf.summary.scalar("scaled_flow_photo_loss",
                                                                   hyp.flow_photo_coeff*flow_photo_loss)
                    scaled_flow_smooth_loss_sum = tf.summary.scalar("scaled_flow_smooth_loss",
                                                                    hyp.flow_smooth_coeff*flow_smooth_loss)
            if hyp.do_odo:
                with tf.name_scope("odo_summ"):
                    if hyp.do_debug:
                        o12_e = tf.check_numerics(o12_e, 'line 341')
                    o12_e_sum = tf.summary.histogram("o12_e", o12_e)
                    o12_g_sum = tf.summary.histogram("o12_g", o12_g)
                    odo_photo_loss_sum = tf.summary.scalar("odo_photo_loss", odo_photo_loss)
                    scaled_odo_photo_loss_sum = tf.summary.scalar("scaled_odo_photo_loss",
                                                                  hyp.odo_photo_coeff*odo_photo_loss)
                    rtd_sum = tf.summary.scalar("rtd", rtd)
                    rta_sum = tf.summary.scalar("rta", rta)
                    td_sum = tf.summary.scalar("td", td)
                    ta_sum = tf.summary.scalar("ta", ta)
                    # scaled_rtd_sum = tf.summary.scalar("scaled_rtd",hyp.rtd_coeff*rtd)
                    # scaled_rta_sum = tf.summary.scalar("scaled_rta",hyp.rta_coeff*rta)
                    # scaled_td_sum = tf.summary.scalar("scaled_td",hyp.td_coeff*td)
                    # scaled_ta_sum = tf.summary.scalar("scaled_ta",hyp.ta_coeff*ta)
            self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def go(self):
        start_time = time.time()
        t_vars = tf.trainable_variables()
        self.d_depth_vars = [var for var in t_vars if 'd_depth' in var.name]
        self.g_depth_vars = [var for var in t_vars if 'DepthNet' in var.name]

        self.d_rt_vars = [var for var in t_vars if 'd_cam' in var.name]
        self.g_rt_vars = [var for var in t_vars if 'Extrinsic' in var.name]

        self.body_vars = [var for var in t_vars if 'd_' not in var.name]
        # self.body_saver = tf.train.Saver(self.body_vars)
        # print "d_depth", [var.name for var in self.d_depth_vars]

        if hyp.do_train:
            print("------ TRAINING ------")
            optimizer = tf.train.AdamOptimizer(hyp.lr, beta1=0.9, beta2=0.999) \
                                .minimize(self.loss, global_step=self.global_step)

            if hyp.do_odo or hyp.do_flow:
                optim_unsup = tf.train.AdamOptimizer(hyp.lr, beta1=0.9, beta2=0.999) \
                                      .minimize(self.unsup_loss, global_step=self.global_step, var_list=self.body_vars)
            if hyp.do_gan:

                g_depth_optim = tf.train.AdamOptimizer(hyp.gan_lr, beta1=0.9, beta2=0.999) \
                                        .minimize(self.g_loss, var_list=self.g_depth_vars)

                d_depth_optim = tf.train.AdamOptimizer(hyp.gan_lr, beta1=0.9, beta2=0.999) \
                                    .minimize(self.d_loss, var_list=self.d_depth_vars)
                fast_d_depth_optim = tf.train.AdamOptimizer(0.00001, beta1=0.9, beta2=0.999)\
                               .minimize(self.d_loss, var_list=self.d_depth_vars) 


        else:
            print("------ TESTING ------")
            if hyp.do_odo:            
                poses_l = open("poses/poses_l_%s.txt" % hyp.name,"w") 
                poses_l.write("rt_dist rt_ang t_ang t_mag\n")
                poses_e = open("poses/poses_e_%s.txt" % hyp.name,"w")
                poses_e.write("frame r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0.1 0.2 1\n")
                poses_g = open("poses/poses_g_%s.txt" % hyp.name,"w")
                poses_g.write("frame r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0.1 0.2 1\n")
        if not hyp.do_unsup:
            writer_t = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)

        writer_v = tf.summary.FileWriter(self.log_dir + '/val', self.sess.graph)
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        if hyp.total_init:
            start_iter = load(self.saver, self.sess, hyp.total_init)
            if start_iter:
                print "loaded full model. resuming from iter %d" % start_iter
            else:
                print "could not find a full model. starting from scratch"
        else:
            if hyp.depth_init:
                depth_start = load_part(self.sess, hyp.depth_init, "depth")
                if depth_start:
                    print "loaded DepthNet at iter %d" % depth_start
                else:
                    print "could not find a DepthNet"
            if hyp.seg_init:
                seg_start = load_part(self.sess, hyp.seg_init, "seg")
                if depth_start:
                    print "loaded SegNet at iter %d" % depth_start
                else:
                    print "could not find a SegNet"
            if hyp.flow_init:
                flow_start = load_part(self.sess, hyp.flow_init, "flow")
                if flow_start:
                    print "loaded FlowNet at iter %d" % flow_start
                else:
                    print "could not find a FlowNet"
            if hyp.odo_init:
                odo_start = load_part(self.sess, hyp.odo_init, "odo")
                if odo_start:
                    print "loaded OdoNet at iter %d" % odo_start
                else:
                    print "could not find an OdoNet"
            # if hyp.gan_init:
            start_iter = 0

        if not hyp.do_train:
            start_iter = 0

        print "OK! Ready to go. (Setup took %.1f)"  % (time.time() - start_time)
        start_time = time.time()
        for step in range(start_iter+1, hyp.max_iters+1):
            # print '\nflag0', step, '\n'
            # on every iteration, get a train batch ...
            train_inputs = self.sess.run([self.i1_g_t,self.i2_g_t,
                                          self.d1_g_t,self.d2_g_t,
                                          self.f12_g_t,
                                          self.v1_g_t,self.v2_g_t,
                                          self.m1_g_t,self.m2_g_t,
                                          self.p1_g_t,self.p2_g_t,
                                          self.s1_g_t, self.s2_g_t,
                                          self.off_h_t,self.off_w_t,
                                          self.fy_t, self.fx_t,
                                          self.y0_t, self.x0_t])
            i1_g_t,i2_g_t,\
                d1_g_t,d2_g_t,\
                f12_g_t,\
                v1_g_t,v2_g_t,\
                m1_g_t,m2_g_t,\
                p1_g_t,p2_g_t,\
                s1_g_t,s2_g_t,\
                off_h_t,off_w_t,\
                fy_t,fx_t,\
                y0_t,x0_t = train_inputs


            # Making gan inputs
            train_inputs_gan = self.sess.run([self.d1_g_n])
            [d1_g_n] = train_inputs_gan

            train_feed = {self.i1_g: i1_g_t, self.i2_g: i2_g_t,
                          self.d1_g: d1_g_t, self.d2_g: d2_g_t,
                          self.f12_g: f12_g_t,
                          self.v1_g: v1_g_t, self.v2_g: v2_g_t,
                          self.m1_g: m1_g_t, self.m2_g: m2_g_t,
                          self.p1_g: p1_g_t, self.p2_g: p2_g_t,
                          self.s1_g : s1_g_t, self.s2_g : s2_g_t,
                          self.off_h: off_h_t, self.off_w: off_w_t,
                          self.fy: fy_t, self.fx: fx_t,
                          self.y0: y0_t, self.x0: x0_t,
                          self.d1_gan:d1_g_n}

            gan_feed = train_feed
            # gan_feed = {self.i1_g: i1_g_t, self.i2_g: i2_g_t,
            #             self.d1_g: d1_g_t, self.d2_g: d2_g_t,
            #             self.f12_g: f12_g_t,
            #             self.v1_g: v1_g_t, self.v2_g: v2_g_t,
            #             self.m1_g: m1_g_t, self.m2_g: m2_g_t,
            #             self.p1_g: p1_g_t, self.p2_g: p2_g_t,
            #             self.s1_g : s1_g_t, self.s2_g : s2_g_t,
            #             self.off_h: off_h_t, self.off_w: off_w_t,
            #             self.fy: fy_t, self.fx: fx_t,
            #             self.y0: y0_t, self.x0: x0_t,
            #             self.d1_gan:d1_g_n}

            # ... and optimize
            if hyp.do_train:

                if hyp.do_gan:
                    d_loss, g_loss, run_sum = self.sess.run([self.d_loss, self.g_loss, self.summary], feed_dict=gan_feed)
                    
                    if d_loss > 0.695 or d_loss > g_loss:
                        _, d_loss, g_loss = \
                          self.sess.run([fast_d_depth_optim, self.d_loss, self.g_loss], feed_dict=gan_feed)       
                    else:
                        _, _, d_loss, g_loss = self.sess.run([d_depth_optim, g_depth_optim, self.d_loss, self.g_loss], feed_dict = gan_feed)
                        if g_loss > 0.75:
                          _, d_loss, g_loss = self.sess.run([g_depth_optim, self.d_loss, self.g_loss], feed_dict=gan_feed)
                        if g_loss > 0.75:
                          _, d_loss, g_loss = self.sess.run([g_depth_optim, self.d_loss, self.g_loss], feed_dict=gan_feed)
                        if g_loss > 0.75:
                          _, d_loss, g_loss = self.sess.run([g_depth_optim, self.d_loss, self.g_loss], feed_dict=gan_feed)
                    # print("GAN: g_depth_loss: %.8f, d_depth_loss:%.8f" %(g_loss, d_loss))


                # NOT TRAINING DEPTH NET 
                if hyp.do_odo or hyp.do_flow:
                    outs = self.sess.run([optim_unsup, optimizer,
                                          self.loss,
                                          self.summary,
                                          self.global_step],
                                         feed_dict=train_feed)
                    _, _, loss, run_sum, step = outs
                else:
                    outs = self.sess.run([optimizer,
                                          self.loss,
                                          self.summary,
                                          self.global_step],
                                         feed_dict=train_feed)
                    _, loss, run_sum, step = outs


                if (step==start_iter+1 or np.mod(step, hyp.log_freq_t) == 0):
                    if not hyp.do_unsup:
                        writer_t.add_summary(run_sum, step)
                    if hyp.do_gan:
                        print("GAN: g_depth_loss: %.8f, d_depth_loss:%.8f" %(g_loss, d_loss))


            # NOT DONE YET FOR GAN
            if not hyp.do_train or (step==start_iter+1 or np.mod(step, hyp.log_freq_v) == 0):
                # on every val iteration, get a val batch ...

                val_inputs = self.sess.run([self.i1_g_v,self.i2_g_v,
                                            self.d1_g_v,self.d2_g_v,
                                            self.f12_g_v,
                                            self.v1_g_v,self.v2_g_v,
                                            self.m1_g_v,self.m2_g_v,
                                            self.p1_g_v,self.p2_g_v,
                                            self.s1_g_v, self.s2_g_v,
                                            self.off_h_v,self.off_w_v,
                                            self.fy_v, self.fx_v,
                                            self.y0_v, self.x0_v])
                i1_g_v,i2_g_v,\
                    d1_g_v,d2_g_v,\
                    f12_g_v,\
                    v1_g_v,v2_g_v,\
                    m1_g_v,m2_g_v,\
                    p1_g_v,p2_g_v,\
                    s1_g_v,s2_g_v,\
                    off_h_v,off_w_v,\
                    fy_v,fx_v,\
                    y0_v,x0_v = val_inputs


                train_inputs_gan = self.sess.run([self.d1_g_g])
                [d1_g_g] = train_inputs_gan

                val_feed = {self.i1_g: i1_g_v, self.i2_g: i2_g_v,
                            self.d1_g: d1_g_v, self.d2_g: d2_g_v,
                            self.f12_g: f12_g_v,
                            self.v1_g: v1_g_v, self.v2_g: v2_g_v,
                            self.m1_g: m1_g_v, self.m2_g: m2_g_v,
                            self.p1_g: p1_g_v, self.p2_g: p2_g_v,
                            self.s1_g : s1_g_v, self.s2_g : s2_g_v,
                            self.off_h: off_h_v, self.off_w: off_w_v,
                            self.fy: fy_v, self.fx: fx_v,
                            self.y0: y0_v, self.x0: x0_v,
                            self.d1_gan:d1_g_g}
                # ... and a variety of plots and visualizations

                if hyp.do_odo:
                    outs = self.sess.run([self.summary,
                                          self.rt12_e,
                                          self.rt12_g,
                                          self.rtd,
                                          self.rta,
                                          self.td,
                                          self.ta],
                                         feed_dict=val_feed)
                    run_sum, \
                        rt12_e, \
                        rt12_g, \
                        rtd, \
                        rta, \
                        td, \
                        ta = outs
                else:
                    outs = self.sess.run([self.summary],
                                         feed_dict=val_feed)
                    run_sum = outs[0]


                writer_v.add_summary(run_sum, step)
                
                if hyp.do_train and hyp.do_odo:
                    print("%s; iter:[%4d/%4d]; time: %.1f; %.2f %.4f %.2f %.2f" \
                          % (hyp.name, step, hyp.max_iters, time.time() - start_time, rtd, rta, td, ta))
                else:
                    print("%s; iter:[%4d/%4d]; time: %.1f" \
                            % (hyp.name, step, hyp.max_iters, time.time() - start_time))

                if not hyp.do_train and hyp.do_odo:
                    poses_l.write('%.6f %.6f %.6f %.6f\n' % (rtd,rta,td,ta))
                    # get the relative transformations too
                    poses_e.write('%d ' % step)
                    poses_e.write((' ').join(("%.6f" % element)
                                             for row in rt12_e[0,:,:]
                                             for element in row))
                    poses_e.write('\n')
                    poses_g.write('%d ' % step)
                    poses_g.write((' ').join(("%.6f" % element)
                                             for row in rt12_g[0,:,:]
                                             for element in row))
                    poses_g.write('\n')
                    print("%s; iter:[%4d/%4d]; time: %.1f; dist: %.4f; angle: %.4f" \
                          % (hyp.name, step, hyp.max_iters, time.time() - start_time, rtd, rta))
            
            if hyp.do_vis and (step==start_iter+1 or np.mod(step, hyp.dump_freq) == 0):
                # on every dump iteration, we'll use the existing val batch
                # ... and get a variety of plots and visualizations
                i1_g_z, i2_g_z, m1_g_z = self.sess.run([self.i1_g_z,
                                                        self.i2_g_z,
                                                        self.m1_g_z],
                                                       feed_dict=val_feed)

                if hyp.do_depth:
                    d1_e_z, d1_g_z = self.sess.run([self.d1_e_z,
                                                    self.d1_g_z],
                                                   feed_dict=val_feed)
                else:
                    d1_e_z, d1_g_z = (None,)*2

                if hyp.do_flow:
                    f12_e_z, f12_g_z, fi1_e_z = self.sess.run([self.f12_e_z,
                                                               self.f12_g_z,
                                                               self.fi1_e_z],
                                                              feed_dict=val_feed)
                    if hyp.do_backward:
                        f21_e_z, fi2_e_z = self.sess.run([self.f21_e_z,
                                                          self.fi2_e_z],
                                                         feed_dict=val_feed)
                    else:
                        f21_e_z, fi2_e_z = (None,)*2
                else:
                    f12_e_z, f12_g_z, fi1_e_z, f21_e_z, fi2_e_z = (None,)*5

                if hyp.do_odo:
                    o12_e_z, o12_g_z, oi1_e_z = self.sess.run([self.o12_e_z,
                                                               self.o12_g_z,
                                                               self.oi1_e_z],
                                                              feed_dict=val_feed)
                    if hyp.do_backward:
                        o21_e_z, o21_g_z, oi2_e_z = self.sess.run([self.o21_e_z,
                                                                   self.o21_g_z,
                                                                   self.oi2_e_z],
                                                                  feed_dict=val_feed)
                    else:
                        o21_e_z, o21_g_z, oi2_e_z = (None,)*3
                else:
                    o12_e_z, o12_g_z, oi1_e_z, o21_e_z, o21_g_z, oi2_e_z = (None,)*6
                   
                dump2disk_gan(self.vis_dir, step,
                          i1=i1_g_z,
                          i2=i2_g_z,
                          d1_e=d1_e_z,
                          d1_g=d1_g_z,
                          f12_e_z=f12_e_z,
                          f12_g_z=f12_g_z,
                          f21_e_z=f21_e_z,
                          fi1_e_z=fi1_e_z,
                          fi2_e_z=fi2_e_z,

                          o12_e_z=o12_e_z,
                          o12_g_z=o12_g_z,
                          o21_e_z=o21_e_z,
                          o21_g_z=o21_g_z,
                          oi1_e_z=oi1_e_z,
                          oi2_e_z=oi2_e_z,
                          m1_g_z=m1_g_z)

                print("%s; dumped visualizations to disk!" % hyp.name)
            # print "\nflag3", step, hyp.snap_freq, "\n"

            if hyp.do_train and (np.mod(step, hyp.snap_freq) == 0):
                save(self.saver, self.sess, self.checkpoint_dir, step)

        if not hyp.do_train and hyp.do_odo:
            poses_l.close()
            poses_e.close()
            poses_g.close()

        coord.request_stop()
        coord.join(threads)
    
    def inference(self, i1_g, i2_g, d1_g, d2_g, f12_g, v1_g, m1_g, rt12_g, s1_g,
                  off_h, off_w, fy, fx, y0, x0, 
                  is_train=True, reuse=False):
        with tf.variable_scope("inference"):
            if hyp.do_depth:
                # CHA adding valid
                depth_stuff = DepthNet(i1_g, d1_g,  hyp.valid, is_train=(is_train and hyp.do_train_depth), reuse=reuse)
            else:
                depth_stuff = 0
            if hyp.do_flow:
                flow_stuff = FlowNet(i1_g, i2_g, f12_g, v1_g, is_train=(is_train and hyp.do_train_flow), reuse=reuse)
            else:
                flow_stuff = 0
            if hyp.do_odo:
                # CHA
                _, d1_e, _ = depth_stuff
                Z1_e = tf.exp(d1_e)
                # Z1_e = tf.stop_gradient(Z1_e)
                Z1_g = tf.exp(d1_g)

                if hyp.do_flow:
                    _, f12_e, _ = flow_stuff
                    f12_e = tf.stop_gradient(f12_e)/50.0 # flow has a pretty high mag
                if hyp.cat_rgb:
                    odo_inputs = tf.concat(3,[i1_g,i2_g])
                    if hyp.do_flow and hyp.cat_flow:
                        odo_inputs = tf.concat(3,[odo_inputs,f12_e])
                else: # we have to use flow
                    odo_inputs = f12_e
                if hyp.cat_angles:
                    ang_xy = tf.expand_dims(angleGrid(hyp.bs, hyp.h, hyp.w, y0, x0),3)
                    odo_inputs = tf.concat(3,[odo_inputs,ang_xy])
                if hyp.do_flow and hyp.cat_ang_flow:
                    x, y = f12_e[:, :, :, 0], f12_e[:, :, :, 1]
                    ang_flow = tf.expand_dims(atan2_ocv(y, x),3)
                    odo_inputs = tf.concat(3,[odo_inputs,ang_flow])
                if hyp.do_flow and hyp.cat_ang_diff:
                    x, y = f12_e[:, :, :, 0], f12_e[:, :, :, 1]
                    ang_flow = tf.expand_dims(atan2_ocv(y, x),3)
                    ang_xy = tf.expand_dims(angleGrid(hyp.bs, hyp.h, hyp.w, y0, x0),3)
                    ang_diff = ang_flow - ang_xy
                    odo_inputs = tf.concat(3,[odo_inputs,ang_diff])
                if hyp.cat_depth:
                    odo_inputs = tf.concat(3,[odo_inputs,Z1_e])
                odo_stuff = OdoNet(odo_inputs,i1_g,i2_g,Z1_e,Z1_g,rt12_g,m1_g,
                                   off_h,off_w,fy,fx,y0,x0,
                                   is_train=(is_train and hyp.do_train_odo), reuse=reuse)
            else:
                odo_stuff = 0

            if hyp.do_seg:
                with tf.variable_scope("seg"):
                    with tf.variable_scope("model"):
                        input_size = (hyp.h, hyp.w)
                        net = DeepLabResNetModel({'data': i1_g}, is_training=hyp.do_train_seg, num_classes=hyp.num_seg_classes)
                        
                        # Predictions.
                        raw_output = net.layers['fc1_voc12']

                        # Predictions: ignoring all predictions with labels greater or equal than n_classes
                        raw_prediction = tf.reshape(raw_output, [-1, hyp.num_seg_classes])
                        label_proc = prepare_label(s1_g, tf.stack(raw_output.get_shape()[1:3]), num_classes=hyp.num_seg_classes, one_hot=False) # [batch_size, h, w]
                        raw_gt = tf.reshape(label_proc, [-1,])
                        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, hyp.num_seg_classes - 1)), 1)
                        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                        prediction = tf.gather(raw_prediction, indices)
                                                                      

                        # Pixel-wise softmax loss.
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
                        
                        # Processed predictions: for visualisation.
                        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(i1_g)[1:3,])
                        raw_output_up_lab = tf.argmax(raw_output_up, dimension=3)
                        pred = tf.expand_dims(raw_output_up_lab, dim=3)

                    # deining the l2 loss on the parameters of the model
                    weight_var = [v for v in tf.trainable_variables() if 'inference/seg/model' in v.name]
                    weight_var = [v for v in weight_var if 'weights' in v.name]
                    
                    # l2_losses = [hyp.seg_weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
                    l2_losses = [hyp.seg_weight_decay * tf.nn.l2_loss(v) for v in weight_var]
                    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
                    seg_stuff = (reduced_loss, raw_output_up, pred)
            else:
                seg_stuff = 0

            return depth_stuff, flow_stuff, odo_stuff, seg_stuff

    
    def discriminator(self, input, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                # CHA in conv2d
            h0 = lrelu(slim.conv2d(input, hyp.df_dim, [5, 5], scope='d_input_h0_conv'))
            h1 = lrelu(self.d_bn1(slim.conv2d(h0, hyp.df_dim*2, [5, 5], scope='d_input_h1_conv')))
            h2 = lrelu(self.d_bn2(slim.conv2d(h1, hyp.df_dim*4, [5, 5], scope='d_input_h2_conv')))
            h3 = slim.conv2d(h2, hyp.df_dim*8, [5, 5], scope='d_input_h3_conv')
            #h4 = linear(tf.reshape(h3, [hyp.bs, -1]), 1, scope='d_input_h4')
        return h3

