import tensorflow as tf
import fcn_utils
from encoder_decoder import *
from losses import *
from surface_normals import surface_normals
from bbox2cuboid import bboxes2cuboids
import sys
from kaffe.tensorflow import Network
from fcn8_vgg import *

import hyperparams_gan as hyp

# sys.path.append('tf_faster_rcnn/tools/')
# sys.path.append('tf_faster_rcnn/lib/model_/')
# sys.path.append('tf_faster_rcnn/lib/')
# sys.path.append('tf_faster_rcnn/lib/nets_')
# from model_.train_val import get_training_roidb, train_net, SolverWrapper
# from model_.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import tensorflow as tf
# from nets_.vgg16 import vgg16
# from nets_.resnet_v1 import resnetv1

'''
sys.path.append('FastMaskRCNN')
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
'''

def DepthNet(im, depth_g, valid, is_train=True, reuse=False):
    with tf.variable_scope("depth"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        depth_stack = encoder_decoder(im, 1, "DepthNet",
                                      hyp.nLayers_depth,
                                      is_train=is_train,
                                      reuse=reuse)
        depth_e = depth_stack[-1]
        if not is_train:
            depth_e = tf.stop_gradient(depth_e)
        depth_m = valid*depth_g+(1-valid)*depth_e

        with tf.variable_scope("loss"):
            l1_main = masked_l1Loss(depth_e, depth_g, valid)
            l1_inval = masked_l1Loss(depth_e, depth_g, 1-valid)
            l1_all = l1Loss(depth_e, depth_g)
            # l1 = 0.5*masked_l1Loss(depth_e, depth_g, valid) + 0.5*l1Loss(depth_e, depth_g)
            smooth = smoothLoss1(depth_e)
            prior = tf.abs(tf.reduce_mean(depth_e)-hyp.depth_prior)
            
            depth_loss = hyp.depth_main_coeff*l1_main + \
                         hyp.depth_inval_coeff*l1_inval + \
                         hyp.depth_smooth_coeff*smooth + \
                         hyp.depth_prior_coeff*prior
            
        with tf.name_scope("depth_summ"):
            d1_e_sum = tf.summary.histogram("d1_e", depth_e)
            d1_g_sum = tf.summary.histogram("d1_g", depth_g)
            d1_e_sum2 = tf.summary.scalar("d1_e", tf.reduce_mean(depth_e))
            d1_g_sum2 = tf.summary.scalar("d1_g", tf.reduce_mean(depth_g))
            l1_main_sum = tf.summary.scalar("l1_main", l1_main)
            scaled_l1_main_sum = tf.summary.scalar("scaled_l1_main",
                                                   hyp.depth_main_coeff*l1_main)
            l1_inval_sum = tf.summary.scalar("l1_inval", l1_inval)
            scaled_l1_inval_sum = tf.summary.scalar("scaled_l1_inval",
                                                  hyp.depth_inval_coeff*l1_inval)
            l1_all_sum = tf.summary.scalar("l1_all", l1_all)
            smooth_sum = tf.summary.scalar("smooth", smooth)
            scaled_smooth_sum = tf.summary.scalar("scaled_smooth",
                                                  hyp.depth_smooth_coeff*smooth)
            prior_sum = tf.summary.scalar("prior", prior)
            scaled_prior_sum = tf.summary.scalar("scaled_prior",
                                                 hyp.depth_prior_coeff*prior)
            
        depth_stuff = [depth_loss, depth_e, depth_m]
    return depth_stuff

def OdoNet(inputs, i1_g, i2_g, Z1_e, Z1_g, rt12_g, static_mask, off_h, off_w, fy, fx, y0, x0,
           is_train=True, reuse=False):
    with tf.variable_scope("odo"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        ## start getting rt12_e
        if hyp.do_fc:
            # when we go fully-connected, we don't need to mask anything 
            pred_stack = encoder_decoder(inputs, 6, "OdoNet",
                                         hyp.nLayers_odo,
                                         is_train=is_train,
                                         do_decode=False, reuse=reuse)
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                rt = slim.fully_connected(tf.reshape(pred_stack[-1],[hyp.bs,-1]), 6)
            r, t = tf.split(1, 2, rt)
        else:
            # when we go convolutional, masks make sense
            pred_stack = encoder_decoder(inputs, 6, "OdoNet",
                                         hyp.nLayers_odo,
                                         is_train=is_train,
                                         do_decode=True, reuse=reuse)
            r, t = tf.split(3, 2, pred_stack[-1])
            if hyp.do_mask_rt:
                keep = tf.tile(static_mask,[1,1,1,3])
                nKeep = tf.reduce_sum(keep,axis=[1,2])
                r_keep = r*keep
                t_keep = t*keep
                r_sum = tf.reduce_sum(r_keep,axis=[1,2])
                t_sum = tf.reduce_sum(t_keep,axis=[1,2])
                r = tf.reshape(r_sum/(nKeep+hyp.eps),[hyp.bs,3])
                t = tf.reshape(t_sum/(nKeep+hyp.eps),[hyp.bs,3])
            else:
                r = tf.reduce_mean(r,axis=[1,2])
                t = tf.reduce_mean(t,axis=[1,2])
        r = tf.minimum(tf.maximum(r*0.001,-1),1)
        [sina, sinb, sing] = tf.unpack(r,axis=1)
        [tx, ty, tz] = tf.unpack(t,axis=1)
        t = tf.pack([tx, ty, tz],axis=1,name="t")
        r = sinabg2r(sina,sinb,sing)
        if not is_train:
            r = tf.stop_gradient(r)
            t = tf.stop_gradient(t)
        rt12_e = merge_rt(r,t)
        ## end getting rt12_e

        o12_e = zrt2flow_helper(Z1_e, rt12_e, fy, fx, y0, x0)
        o12_g = zrt2flow_helper(Z1_g, rt12_g, fy, fx, y0, x0)
        i1_e, i2_nocc = warper(i2_g, o12_e)
        oi1_g, _ = warper(i2_g, o12_g)
        with tf.variable_scope("loss"):
            if hyp.do_mask_photo:
                photo = masked_l1Loss(i1_e,i1_g,static_mask)
            else:
                photo = l1Loss(i1_e,i1_g)
            rt_loss = rtLoss(rt12_e,rt12_g)
            rtd, rta, td, ta = rt_loss
            odo_loss = hyp.odo_photo_coeff*photo + \
                       hyp.rtd_coeff*rtd + \
                       hyp.rta_coeff*rta + \
                       hyp.td_coeff*td + \
                       hyp.ta_coeff*ta

        with tf.name_scope("odo_summ"):
            if hyp.do_debug:
                o12_e = tf.check_numerics(o12_e, 'line 341')
            o12_e_sum = tf.summary.histogram("o12_e", o12_e)
            o12_g_sum = tf.summary.histogram("o12_g", o12_g)
            fy_sum = tf.summary.histogram("fy", fy)
            fx_sum = tf.summary.histogram("fx", fx)
            y0_sum = tf.summary.histogram("y0", y0)
            x0_sum = tf.summary.histogram("x0", x0)
            photo_sum = tf.summary.scalar("photo", photo)
            scaled_photo_sum = tf.summary.scalar("scaled_photo", hyp.odo_photo_coeff*photo)

            rtd_sum = tf.summary.scalar("rtd", rtd)
            rta_sum = tf.summary.scalar("rta", rta)
            td_sum = tf.summary.scalar("td", td)
            ta_sum = tf.summary.scalar("ta", ta)
            scaled_rtd_sum = tf.summary.scalar("scaled_rtd", hyp.rtd_coeff*rtd)
            scaled_rta_sum = tf.summary.scalar("scaled_rta", hyp.rta_coeff*rta)
            scaled_td_sum = tf.summary.scalar("scaled_td", hyp.td_coeff*td)
            scaled_ta_sum = tf.summary.scalar("scaled_ta", hyp.ta_coeff*ta)
        odo_stuff = [odo_loss, rt_loss, rt12_e, o12_e, o12_g, i1_e, oi1_g]
    return odo_stuff

def FlowNet(i1_g, i2_g, f12_g, v12, is_train=True, reuse=False):
    with tf.variable_scope("flow"):
        concat = tf.concat(3, [i1_g, i2_g], name="concat")
        pred_stack = encoder_decoder(concat, 2, "FlowNet",
                                     hyp.nLayers_flow,
                                     is_train=is_train,
                                     do_decode=True, reuse=reuse)
        f12_e = pred_stack[-1]*20
        if not is_train:
            f12_e = tf.stop_gradient(f12_e)
        i1_e, _ = warper(i2_g, f12_e)
        with tf.variable_scope("loss"):
            l2 = masked_l2Loss(f12_e, f12_g, v12)
            # l2 = l2Loss(f12_e, f12_g)
            photo = l1Loss(i1_e,i1_g)
            smooth = smoothLoss2(f12_e)
            # CHA
            flow_loss = (l2, photo, smooth)
        if hyp.do_debug:
            f12_e = tf.check_numerics(f12_e, 'line 947')
        with tf.name_scope("summ"):
            f12_e_sum = tf.summary.histogram("f12_e", f12_e)
            f12_g_sum = tf.summary.histogram("f12_g", f12_g)
            l2_sum = tf.summary.scalar("l2", l2)
            photo_sum = tf.summary.scalar("photo", photo)
            smooth_sum = tf.summary.scalar("smooth", smooth)
            scaled_l2_sum = tf.summary.scalar("scaled_l2", hyp.flow_l2_coeff*l2)
            scaled_photo_sum = tf.summary.scalar("scaled_photo", hyp.flow_photo_coeff*photo)
            scaled_smooth_sum = tf.summary.scalar("scaled_smooth", hyp.flow_smooth_coeff*smooth)
        flow_stuff = [flow_loss, f12_e, i1_e]
    return flow_stuff
        
def MaskNet(frame, toy_g, tab_g, is_train=True, reuse=False):
    with tf.variable_scope("mask"):
        pred_stack = encoder_decoder(frame, 1, "MaskNet",
                                     hyp.nLayers_mask,
                                     is_train=is_train,
                                     reuse=reuse)
        mask = pred_stack[-1]
        mask = tf.sigmoid(mask)
        if not is_train:
            mask = tf.stop_gradient(mask)
        mask_merged = toy_g+(1-toy_g)*mask
            
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(mask,toy_g)
        # only include the pixels for which we have labels
        have = toy_g+tab_g
        cross_entropy = cross_entropy*have
        cross_entropy = tf.reduce_sum(cross_entropy,axis=[1,2])/tf.reduce_sum(have,axis=[1,2])
        cross_entropy = tf.reduce_mean(cross_entropy)
        smooth = smoothLoss1(mask)
        mask_loss = hyp.mask_coeff*cross_entropy + \
                    hyp.mask_smooth_coeff*smooth
        mask_stuff = [mask_loss, mask, mask_merged, toy_g, tab_g]
        
        with tf.name_scope("summ"):
            cross_entropy_sum = tf.summary.scalar("cross_entropy", cross_entropy)
            scaled_cross_entropy_sum = tf.summary.scalar("scaled_cross_entropy",
                                                         hyp.mask_coeff*cross_entropy)
            smooth_loss_sum = tf.summary.scalar("smooth", smooth)
            scaled_smooth_loss_sum = tf.summary.scalar("scaled_smooth",
                                                       hyp.mask_smooth_coeff*smooth)
    return mask_stuff

def PoseNet(inputs, i1_g, i2_g, Z1, Z2, toy_mask,fy,fx,is_train=True, reuse=False):
    with tf.variable_scope("pose"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # we need a pivot point for the rt, but the absolute pivot would
        # probably be super hard to estimate, especially convolutionally.
        # instead, we estimate the delta pivot, dp
            
        # let's collect both things:
        # 1) warping each point with a unique [R,t]
        # 2) warping everything with the [R,t] averaged over the toy mask

        pred_stack = encoder_decoder(inputs, 9, "PoseNet",
                                     hyp.nLayers_pose,
                                     is_train=is_train,
                                     reuse=reuse)
        
        sine = tf.slice(pred_stack[-1], [0,0,0,0], [-1,-1,-1,3])
        transl = tf.slice(pred_stack[-1], [0,0,0,3], [-1,-1,-1,3])
        dp = tf.slice(pred_stack[-1], [0,0,0,6], [-1,-1,-1,3])

        # rotation sines are powerful transformations.
        # let's scale them down, and make sure they're within [0,1]
        sine = tf.minimum(tf.maximum(sine*0.001,-1),1)

        # first, we can just collect these guys across the whole image
        sine = tf.reshape(sine,[hyp.bs,-1,3])
        transl = tf.reshape(transl,[hyp.bs,-1,3])
        dp = tf.reshape(dp,[hyp.bs,-1,3])
        [sina, sinb, sing] = tf.unpack(sine,axis=2)
        [tx, ty, tz] = tf.unpack(transl,axis=2)
        translmat = tf.pack([tx, ty, tz],axis=2)
        rotmat = sinabg2r_fc(sina,sinb,sing)

        # we can also collect them within the toy mask. this is what we care about.
        nKeep = tf.reduce_sum(toy_mask,axis=[1,2])
        toy_mask_flat = tf.reshape(tf.tile(toy_mask,[1,1,1,3]),[hyp.bs,-1,3])
        dp_keep = dp*toy_mask_flat
        sine_keep = sine*toy_mask_flat
        transl_keep = transl*toy_mask_flat
        sine_sum = tf.reduce_sum(sine_keep,axis=1)
        transl_sum = tf.reduce_sum(transl_keep,axis=1)
        dp_sum = tf.reduce_sum(dp_keep,axis=1)
        sine_avg = tf.reshape(sine_sum/(nKeep+hyp.eps),[hyp.bs,3],name="sine_avg")
        [sina, sinb, sing] = tf.unpack(sine_avg,axis=1)
        rotmat2 = sinabg2r(sina,sinb,sing)
        translmat2 = tf.reshape(transl_sum/(nKeep+hyp.eps),[hyp.bs,3],name="transl_avg")
        dp2 = tf.reshape(dp_sum/(nKeep+hyp.eps),[hyp.bs,3],name="dp_avg")
        # though these are averages, we want them per-pixel
        dp2 = tf.tile(tf.expand_dims(dp2,1),[1,hyp.h*hyp.w,1])
        rotmat2 = tf.tile(tf.expand_dims(rotmat2,1),[1,hyp.h*hyp.w,1,1])
        translmat2 = tf.tile(tf.expand_dims(translmat2,1),[1,hyp.h*hyp.w,1])
        
        p12_e, u, v, XYZ1_transformed = zdrt2flow_fc(Z1,dp,rotmat,translmat,hyp.scale,fy,fx)
        i1_e, _ = warper(i2_g, p12_e)
        p12_e2, u2, v2, XYZ1_transformed2 = zdrt2flow_fc(Z1,dp2,rotmat2,translmat2,hyp.scale,fy,fx)
        i1_e2, _ = warper(i2_g, p12_e2)

        # XYZ1_transformed(2) is an estimate of XYZ2
        [_,_,Z1_transformed] = tf.split(2, 3, XYZ1_transformed2)
        Z1_transformed = tf.reshape(Z1_transformed,[hyp.bs,hyp.h,hyp.w,1])
        # we want to check if Z2 and Z1_transformed are similar
        # but first we have to warp Z2 to the coordinate frame of Z1
        Z2_e = tf.reshape(Z2,[hyp.bs,hyp.h,hyp.w,1])
        Z2_e_warped, _ = warper(Z2_e, p12_e2)
        depthwarp = l1Loss(Z1_transformed, Z2_e_warped)
        mdepthwarp = masked_l1Loss(Z1_transformed, Z2_e_warped, toy_mask)
        
        with tf.variable_scope("rt_4x4"):
            # bottom_row = tf.tile(tf.reshape(tf.pack([0.,0.,0.,1.]),[1,1,4]),
            #                      [hyp.bs,1,1],name="bottom_row")
            rt12_e = tf.concat(2,[rotmat,tf.expand_dims(translmat,2)],name="rt12_3x4")
            # rt12_e = tf.concat(1,[rt12_e,bottom_row],name="rt12_4x4")
        with tf.variable_scope("loss"):
            [dpx, dpy, dpz] = tf.split(3, 3, tf.reshape(dp,[hyp.bs,hyp.h,hyp.w,3]))
            [tx12, ty12, tz12] = tf.split(3, 3, tf.reshape(translmat,[hyp.bs,hyp.h,hyp.w,3]))
            [sa, sb, sg] = tf.split(3, 3, tf.reshape(sine,[hyp.bs,hyp.h,hyp.w,3]))

            # print '&'*50
            print_shape(i1_e)
            print_shape(i1_g)
            print_shape(toy_mask)
            
            photo = l1Loss(i1_e,i1_g)
            photo2 = l1Loss(i1_e2,i1_g)
            mphoto = masked_l1Loss(i1_e,i1_g,toy_mask)
            mphoto2 = masked_l1Loss(i1_e2,i1_g,toy_mask)

            smooth_dp = smoothLoss1(dpx)+smoothLoss1(dpy)+smoothLoss1(dpz)
            smooth_sine = smoothLoss1(sa)+smoothLoss1(sb)+smoothLoss1(sg)
            msmooth_dp = masked_smoothLoss1(dpx,toy_mask)+masked_smoothLoss1(dpy,toy_mask)+masked_smoothLoss1(dpz,toy_mask)
            msmooth_sine = masked_smoothLoss1(sa,toy_mask)+masked_smoothLoss1(sb,toy_mask)+masked_smoothLoss1(sg,toy_mask)
            
            pose_loss = hyp.pose_photo_coeff*photo + \
                        hyp.pose_photo2_coeff*photo2 + \
                        hyp.pose_mphoto_coeff*mphoto + \
                        hyp.pose_mphoto2_coeff*mphoto2 + \
                        hyp.pose_smooth_dp_coeff*smooth_dp + \
                        hyp.pose_smooth_sine_coeff*smooth_sine + \
                        hyp.pose_msmooth_dp_coeff*msmooth_dp + \
                        hyp.pose_msmooth_sine_coeff*msmooth_sine + \
                        hyp.pose_depthwarp_coeff*depthwarp + \
                        hyp.pose_mdepthwarp_coeff*mdepthwarp

        with tf.name_scope("pose_summ"):
            pose_loss_sum = tf.summary.scalar("pose_loss", pose_loss)

            p12_e_sum = tf.summary.histogram("p12_e", p12_e)
            p12_e2_sum = tf.summary.histogram("p12_e2", p12_e2)

            photo_sum = tf.summary.scalar("photo", photo)
            photo2_sum = tf.summary.scalar("photo2", photo2)
            scaled_photo_sum = tf.summary.scalar("scaled_photo", hyp.pose_photo_coeff*photo)
            scaled_photo2_sum = tf.summary.scalar("scaled_photo2", hyp.pose_photo2_coeff*photo2)

            mphoto_sum = tf.summary.scalar("mphoto", mphoto)
            mphoto2_sum = tf.summary.scalar("mphoto2", mphoto2)
            scaled_mphoto_sum = tf.summary.scalar("scaled_mphoto", hyp.pose_mphoto_coeff*mphoto)
            scaled_mphoto2_sum = tf.summary.scalar("scaled_mphoto2", hyp.pose_mphoto2_coeff*mphoto2)

            smooth_dp_sum = tf.summary.scalar("smooth_dp", smooth_dp)
            scaled_smooth_dp_sum = tf.summary.scalar("scaled_smooth_dp", hyp.pose_smooth_dp_coeff*smooth_dp)
            smooth_sine_sum = tf.summary.scalar("smooth_sine", smooth_sine)
            scaled_smooth_sine_sum = tf.summary.scalar("scaled_smooth_sine", hyp.pose_smooth_sine_coeff*smooth_sine)

            msmooth_dp_sum = tf.summary.scalar("msmooth_dp", msmooth_dp)
            scaled_msmooth_dp_sum = tf.summary.scalar("scaled_msmooth_dp", hyp.pose_msmooth_dp_coeff*msmooth_dp)
            msmooth_sine_sum = tf.summary.scalar("msmooth_sine", msmooth_sine)
            scaled_msmooth_sine_sum = tf.summary.scalar("scaled_msmooth_sine", hyp.pose_msmooth_sine_coeff*msmooth_sine)

            depthwarp_sum = tf.summary.scalar("depthwarp", depthwarp)
            scaled_depthwarp_sum = tf.summary.scalar("scaled_depthwarp", hyp.pose_depthwarp_coeff*depthwarp)
            mdepthwarp_sum = tf.summary.scalar("mdepthwarp", mdepthwarp)
            scaled_mdepthwarp_sum = tf.summary.scalar("scaled_mdepthwarp", hyp.pose_mdepthwarp_coeff*mdepthwarp)

        pose_stuff = [pose_loss, rt12_e, p12_e, p12_e2, i1_e, i1_e2, dp, dp2]
    return pose_stuff
    
def ComNet(i1_g, i2_g, Z1, Z2, m1, m2, fy, fx, y0, x0, is_train=True, reuse=False):
    # this network is in charge of finding the center of mass (com) of an object,
    # given the object's image.

    # we will not estimate the com directly.
    # instead, we will esimate the 3D delta from the current location,
    # using the pixel grid and the estimated depth map as (x,y,z)
    
    with tf.variable_scope("com"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # let's collect both things:
        # 1) com estimated at each pixel
        # 2) com estimated using the average within the toy mask

        def get_com(image, mask, is_train=True, reuse=False):
            pred_stack = encoder_decoder(image, 3, "ComNet",
                                         hyp.nLayers_com,
                                         is_train=is_train,
                                         reuse=reuse)
            com = pred_stack[-1]

            with tf.name_scope("com_summ"):
                com_x, com_y, com_z = tf.split(3,3,com)
                com_x_sum = tf.summary.histogram("com_x", com_x)
                com_y_sum = tf.summary.histogram("com_y", com_y)
                com_z_sum = tf.summary.histogram("com_z", com_z)
            
            # com = com + grid_xyz
            com = tf.reshape(com,[hyp.bs,-1,3])

            # average the coms within toy mask
            nKeep = tf.reduce_sum(mask,axis=[1,2])
            mask_flat = tf.reshape(tf.tile(mask,[1,1,1,3]),[hyp.bs,-1,3])
            com_keep = com*mask_flat
            com_sum = tf.reduce_sum(com_keep,axis=1)
            com_mavg = tf.reshape(com_sum/(nKeep+hyp.eps),[hyp.bs,3],name="com_avg")
            # tile the average to fill the image
            com_mavg = tf.tile(tf.expand_dims(com_mavg,1),[1,hyp.h*hyp.w,1])
            return com, com_mavg

        # we are constructing the motion just by estimating the com twice.
        com1, mcom1 = get_com(i1_g, m1, is_train=is_train, reuse=reuse)
        com2, mcom2 = get_com(i2_g, m2, is_train=is_train, reuse=True)
        
        # com12_e, XYZ1_transformed = zcom2flow_fc(Z1,mcom1,mcom2,hyp.scale,fy,fx)
        # com12_e, XYZ1_transformed,  xy_transformed_flat = zcom2flow_fc(Z1,com1,com2,hyp.scale,fy,fx)
        com12_e, XYZ1_transformed,  xy_transformed_flat = zcom2flow_fc(Z1,mcom1,mcom2,hyp.scale,fy,fx)
        
        i1_e, _ = warper(i2_g, com12_e)

        # ok, visualizing is still a bit tricky, so let's do the loss first
        # i'm estimating -- per pixel -- what XYZ to add, so that the point is at the COM

        com1_e, _ = zcom2com_fc(Z1,com1,fy,fx)
        com2_e, _ = zcom2com_fc(Z2,com2,fy,fx)
        ti1_e, _ = warper(i1_g, com1_e)
        # tm1, _ = warper(m1, com1_e)
        ti2_e, _ = warper(i2_g, com2_e)

        # loss "": be within the object
        # loss 0: make the com arrows sum to 0
        # c_loss = tf.reduce_mean(tf.abs(com1))
        # loss 1: reduce the spread, after applying the com arrows
        
        centroid_loss = com_centroid_loss(Z1, com1, m1, fy, fx)
        spread_loss = com_spread_loss(Z1, com1, m1, fy, fx)
        zerosum_loss = tf.abs(tf.reduce_mean(com1))
        
        # # xy_transformed_flat
        # [xt_flat,yt_flat]=tf.split(2,2,xy_transformed_flat)

        mean_x = 0
        mean_y = 0
        
        # depthwarp loss
        [_,_,Z1_transformed] = tf.split(2, 3, XYZ1_transformed)
        Z1_transformed = tf.reshape(Z1_transformed,[hyp.bs,hyp.h,hyp.w,1])
        # we want to check if Z2 and Z1_transformed are similar
        # but first we have to warp Z2 to the coordinate frame of Z1
        Z2_e = tf.reshape(Z2,[hyp.bs,hyp.h,hyp.w,1])
        Z2_e_warped, _ = warper(Z2_e, com12_e)
        depthwarp = l1Loss(Z1_transformed, Z2_e_warped)
        mdepthwarp = masked_l1Loss(Z1_transformed, Z2_e_warped, m1)
        
        with tf.variable_scope("loss"):
            [com1x, com1y, com1z] = tf.split(3, 3, tf.reshape(com1,[hyp.bs,hyp.h,hyp.w,3]))

            photo = l1Loss(i1_e,i1_g)
            # photo2 = l1Loss(i1_e_masked,i1_g)
            photo2 = 0
            mphoto = masked_l1Loss(i1_e,i1_g,m1)
            mphoto2 = 0
            # mphoto2 = masked_l1Loss(i1_e_masked,i1_g,m1)

            smooth = smoothLoss1(com1x)+smoothLoss1(com1y)+smoothLoss1(com1z)
            msmooth = masked_smoothLoss1(com1x,m1)+masked_smoothLoss1(com1y,m1)+masked_smoothLoss1(com1z,m1)
            
            com_loss = hyp.com_photo_coeff*photo + \
                       hyp.com_photo2_coeff*photo2 + \
                       hyp.com_mphoto_coeff*mphoto + \
                       hyp.com_mphoto2_coeff*mphoto2 + \
                       hyp.com_smooth_coeff*smooth + \
                       hyp.com_msmooth_coeff*msmooth + \
                       hyp.com_depthwarp_coeff*depthwarp + \
                       hyp.com_mdepthwarp_coeff*mdepthwarp + \
                       mean_x + mean_y + \
                       hyp.com_centroid_coeff*centroid_loss + \
                       hyp.com_spread_coeff*spread_loss + \
                       hyp.com_zerosum_coeff*zerosum_loss

        with tf.name_scope("com_summ"):
            com_loss_sum = tf.summary.scalar("com_loss", com_loss)

            c12_e_sum = tf.summary.histogram("c12_e", com12_e)
            # c12_e_masked_sum = tf.summary.histogram("c12_e_masked", c12_e_masked)

            centroid_sum = tf.summary.scalar("centroid", centroid_loss)
            scaled_centroid_sum = tf.summary.scalar("scaled_centroid", hyp.com_centroid_coeff*centroid_loss)
            spread_sum = tf.summary.scalar("spread", spread_loss)
            scaled_spread_sum = tf.summary.scalar("scaled_spread", hyp.com_spread_coeff*spread_loss)
            zerosum_sum = tf.summary.scalar("zerosum", zerosum_loss)
            scaled_zerosum_sum = tf.summary.scalar("scaled_zerosum", hyp.com_zerosum_coeff*zerosum_loss)
            
            photo_sum = tf.summary.scalar("photo", photo)
            photo2_sum = tf.summary.scalar("photo2", photo2)
            scaled_photo_sum = tf.summary.scalar("scaled_photo", hyp.com_photo_coeff*photo)
            scaled_photo2_sum = tf.summary.scalar("scaled_photo2", hyp.com_photo2_coeff*photo2)

            mphoto_sum = tf.summary.scalar("mphoto", mphoto)
            mphoto2_sum = tf.summary.scalar("mphoto2", mphoto2)
            scaled_mphoto_sum = tf.summary.scalar("scaled_mphoto", hyp.com_mphoto_coeff*mphoto)
            scaled_mphoto2_sum = tf.summary.scalar("scaled_mphoto2", hyp.com_mphoto2_coeff*mphoto2)

            smooth_sum = tf.summary.scalar("smooth", smooth)
            scaled_smooth_sum = tf.summary.scalar("scaled_smooth", hyp.com_smooth_coeff*smooth)

            msmooth_sum = tf.summary.scalar("msmooth", msmooth)
            scaled_msmooth_sum = tf.summary.scalar("scaled_msmooth", hyp.com_msmooth_coeff*msmooth)

            depthwarp_sum = tf.summary.scalar("depthwarp", depthwarp)
            scaled_depthwarp_sum = tf.summary.scalar("scaled_depthwarp", hyp.com_depthwarp_coeff*depthwarp)
            mdepthwarp_sum = tf.summary.scalar("mdepthwarp", mdepthwarp)
            scaled_mdepthwarp_sum = tf.summary.scalar("scaled_mdepthwarp", hyp.com_mdepthwarp_coeff*mdepthwarp)

        com_stuff = [com_loss, com12_e, i1_e, com1, com2, ti1_e, ti2_e]
        # tm1_e = tf.tile(tm1,[1,1,1,3])
        # tm2_e = tf.tile(tm2,[1,1,1,3])
        # com_stuff = [com_loss, com12_e, i1_e, com1, com2, tm1, tm2]
    return com_stuff
    
#not technically a network
def ObjectNet((counts, labels, bboxes, poses, masks), depth_e, off_h, off_w):

    def reduce_min_(X):
        #reduce min, but values less than 0 not taken into account
        lz = tf.cast(tf.less(X, 1.0), tf.float32)
        X += lz*1E10
        Y = tf.reduce_min(X, axis = 0)
        le = tf.cast(tf.less(Y, 1E9), tf.float32)
        Y *= le
        return Y

    #returns depth as estimated by all the objects in the image and their sizes
    #what if num_objs is 0?
    def get_depth((num_objs, obj_id, bbox2d, pose3d, mask, 
                   off_h_single, off_w_single), GTDEPTH = True):

        xyz = tf.slice(pose3d, [0, 0], [-1, 3])
        whl = tf.slice(pose3d, [0, 3], [-1, 3])
        rxryrz = tf.slice(pose3d, [0, 6], [-1, 3])
        x = tf.slice(xyz, [0,0], [-1,1])
        y = tf.slice(xyz, [0,1], [-1,1])
        z = tf.slice(xyz, [0,2], [-1,1])

        if GTDEPTH:
            # depth = tf.sqrt(x*x+y*y+z*z+hyp.eps) 
            depth = z
        elif not hyp.do_bbox2cuboid: 
            #we want to estimate depth based on the object class.
            #both classes have about the same size anyway...
            sizeprior = np.array([1.61, 1.55, 3.9])
            avgsize = np.prod(sizeprior)**(1.0/3.0)

            _f = lambda x : tf.cast(x, tf.float32)
            Ls = _f(tf.slice(bbox2d, [0,0], [-1,1]))
            Ts = _f(tf.slice(bbox2d, [0,1], [-1,1]))
            Rs = _f(tf.slice(bbox2d, [0,2], [-1,1]))
            Bs = _f(tf.slice(bbox2d, [0,3], [-1,1]))
            
            bboxsize = tf.sqrt(tf.abs(Ls-Rs)*tf.abs(Bs-Ts)+hyp.eps)
            depth = hyp.fx*hyp.scale*avgsize/bboxsize
        else: #estimate depth with bbox2cuboid
            dimensions = tf.expand_dims(tf.constant([1.61, 1.55, 3.9]), 0)
            dimensions = tf.tile(dimensions, tf.pack([tf.shape(pose3d)[0], 1]))
            rots = poses2rots(pose3d)
            
            XYZs = bboxes2cuboids(bbox2d, rots, dimensions, 
                                  tf.constant(hyp.fx), 
                                  tf.constant(hyp.fy), 
                                  tf.constant(hyp.x0), 
                                  tf.constant(hyp.y0), 
                                  off_h_single, off_w_single)
            Xs, Ys, Zs = XYZs
            depth = tf.sqrt(Xs*Xs+Ys*Ys+Zs*Zs+hyp.eps)

        #we alread have the mask
        obj_mask = mask
        #zeroes everywhere except within bbox2d
        #obj_mask = makemask(bbox2d) #this givyes us a stack of masks
        valid_mask = tf.reduce_max(obj_mask, axis = 0)
        tiledepth = tf.tile(tf.expand_dims(depth, axis = 2), (1, hyp.h, hyp.w))
        obj_depths = obj_mask*tiledepth

        #trim by num_objs
        padsize = tf.cast(tf.shape(z)[0], tf.int64)
        diff = tf.cast(padsize-num_objs, tf.int64)
        shape_ = tf.pack([tf.cast(diff, tf.int32), hyp.h, hyp.w])
        B = tf.ones(shape = shape_)
        shape__ = tf.pack([tf.cast(num_objs, tf.int32), hyp.h, hyp.w])
        A = tf.ones(shape = shape__)
        multiplier = tf.concat(0, [A,B])

        #we don't really want to max here, because of overlap
        obj_depths = reduce_min_(obj_depths*multiplier)
        
        #_obj_depths = tf.ones((hyp.h, hyp.w))
        #_valid_mask = tf.zeros((hyp.h, hyp.w))
        obj_depths, valid_mask = tf.cond(tf.greater(num_objs, 0), 
                                         lambda: [obj_depths, valid_mask], 
                                         lambda: [tf.ones((hyp.h, hyp.w)), 
                                                  tf.zeros((hyp.h, hyp.w))])

        return obj_depths, valid_mask

    with tf.variable_scope("object"):
        counts = tf.cast(counts, tf.int64)
        off_h = tf.cast(off_h, tf.float32)
        off_w = tf.cast(off_w, tf.float32)

        #obj_depths = tf.reduce_min(mask, axis = 0) #this better work...
        #valid_mask = tf.ones_like(obj_depths)

        #we have one depth map per image in the batch
        depth_maps, valid_masks = tf.map_fn(get_depth, 
                                            [counts, labels, bboxes, poses, masks, 
                                             off_h, off_w], 
                                            dtype = (tf.float32, tf.float32))

        depth_maps = encode_depth(depth_maps)

        with tf.variable_scope("loss"):
            # hinge_loss = 0
            hinge_loss = masked_hingeloss(depth_e, 
                                          tf.expand_dims(depth_maps, 3), 
                                          tf.expand_dims(valid_masks, 3))
            object_loss = hyp.object_depth_rule_coeff*hinge_loss
        with tf.name_scope("object_summ"):
            hinge_sum = tf.summary.scalar("hinge", hinge_loss)
            scaled_hinge_sum = tf.summary.scalar("scaled_hinge",
                                                 hyp.object_depth_rule_coeff*hinge_loss)
    return object_loss, depth_maps, valid_masks

def Normals(encoded_depth, seg):
    sn = surface_normals(encoded_depth)
    roadmask = tf.cast(tf.equal(seg, 6), tf.float32)
    smooth_loss = masked_smoothLoss_MC(sn, roadmask)
    up = tf.reshape([1.0,0.0,0.0], [1, 1, 1, 3]) #yxz
    masksize = tf.reduce_sum(roadmask)+hyp.eps
    road_l2 = tf.sqrt(tf.reduce_sum(tf.square(sn-up), axis = 3))
    roadup_loss = tf.reduce_sum(tf.squeeze(roadmask)*road_l2)/masksize
    loss = smooth_loss*hyp.normals_smooth_coeff+roadup_loss*hyp.normals_roadup_coeff
    return sn, loss

#sets up the mask rcnn
def MaskRCNN(nCars, carseg, image):
    gt_masks, gt_boxes = seg2masksandboxes(nCars, carseg)
    logits, end_points, pyramid_map = network.get_network('resnet50', image)
    outputs = pyramid_network.build(end_points, hyp.h, hyp.w, pyramid_map, 
                                    num_classes=3, 
                                    base_anchors=9,
                                    is_training=True,
                                    gt_boxes=gt_boxes, 
                                    gt_masks=gt_masks,
                                    loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])
    total_loss = outputs['total_loss']
    loss  = outputs['losses']
    batch_info = outputs['batch_info']

    #somehow extract predicted bboxes, poses, labels, and count from outputs
    count = outputs['count']
    labels = outputs['pred_classes']
    bboxes = outputs['pred_boxes']
    masks = outputs['pred_masks']
    poses = outputs['pred_poses']

    out = count, labels, bboxes, poses, masks
    return out, loss

def FasterRCNN(img, sess, objstuff, carstuff):
    '''
    todo list:
    1. remove inactive bounding boxes from each frame somwhow
    2. put loss on pred mask
    3. go from bbox+ predicted mask to image mask
    '''
    
    nc = 2 #2 classes

    net = resnetv1(batch_size = hyp.bs, num_layers = 50)

    '''
    im_info = makeinfo(carstuff[0])
    gt_boxes = tf.squeeze(extract_gt_boxes(carstuff)) #in utils
    '''
    im_info = tf.constant([[hyp.h, hyp.w, 1.0]])

    if hyp.gt_box_source == 'objs':
        gt_boxes = tf.cast(tf.squeeze(objstuff[2], axis = 0), tf.float32)
        gt_boxes = tf.pad(gt_boxes, [[0,0],[0,1]], mode='CONSTANT')
    elif hyp.gt_box_source == 'segs':
        #this is not complete!!!
        gt_boxes = extract_gt_boxes(carstuff)
        gt_boxes = tf.cast(gt_boxes, tf.float32)
    else:
        assert False

    layers = net.create_architecture(sess, 'TRAIN', nc, 
                                     img, 
                                     im_info,
                                     gt_boxes,
                                     tag='default',
                                     anchor_scales=cfg.ANCHOR_SCALES,
                                     anchor_ratios=cfg.ANCHOR_RATIOS)
    
    cls_prob = layers['cls_prob']
    cls_idx = tf.argmax(cls_prob, axis = 1) #128
    mask = layers['mask_pred']
    mask = select_by_last(cls_idx, mask) #128x14x14

    #binarize this mask and make it to the correct shape!

    pose = layers['pose_pred']
    pose = tf.reshape(pose, (-1, 3, nc))
    pose = select_by_last(cls_idx, pose)

    #no loss on size yet
    size = layers['size_pred']
    size = tf.reshape(size, (-1, 3, nc))
    size = select_by_last(cls_idx, size)

    bbox = layers['bbox_pred']
    bbox = tf.reshape(bbox, (-1, 4, nc))
    bbox = select_by_last(cls_idx, bbox)

    numobjs = tf.shape(cls_idx)[0] #= 128

    #for name, tensor in layers.items():
    #    print name, tensor

    masks = makemask(bbox) #replace with predicted mask later

    things = [numobjs, cls_idx, bbox, pose, masks]
    loss = layers['total_loss']

    return things, loss

def ObjectCounter((counts, labels, bboxes, poses, masks), depth_e, off_h, off_w):
    counts = tf.cast(counts, tf.int64)
    counts = tf.Print(counts, [counts], message="This is the object count: ")
    return (counts, labels, bboxes, poses, masks)


# def SemSeg(i1_g, s1_g):
#         #returns prediction and loss
#         segfcn = FCN8VGG()
#         segfcn.build(i1_g,
#                      train = hyp.do_train_seg,
#                      num_classes = hyp.num_seg_classes,
#                      random_init_fc8=True)

#         # B x H x W x 14
#         s1_e = segfcn.pred_up
#         s1_g = tf.reshape(s1_g, tf.shape(s1_g)[:-1])
#         print_shape(s1_e)
#         print_shape(s1_g)
        
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.subtract(s1_g, tf.ones_like(s1_g, dtype=tf.int32)) , logits = segfcn.upscore32)
#         loss = tf.reduce_mean(loss, name = "seg_loss")

#         seg_loss = loss * hyp.seg_coeff

#         with tf.name_scope("seg_summ"):
#             seg_loss_sum = tf.summary.scalar("seg_loss", seg_loss)
#             seg_g_image = tf.summary.image("seg_g_image", tf.to_float(tf.expand_dims(s1_g, 3)))
#             seg_e_image = tf.summary.image("seg_e_image", tf.to_float(tf.expand_dims(s1_e, 3)))
#             seg_i = tf.summary.image("seg_image", i1_g)
#         return [segfcn.upscore32, segfcn.pred_up, seg_loss]

class DeepLabResNetModel(Network):
    def setup(self, is_training, num_classes):
        with tf.variable_scope("Resnet") as resnet_scope:
            # print resnet_scope

            '''Network definition.
            Args:
              is_training: whether to update the running mean and variance of the batch normalisation layer.
                           If the batch size is small, it is better to keep the running mean and variance of 
                           the-pretrained model frozen.
              num_classes: number of classes to predict (including background).
            '''

            (self.feed('data')
                 .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
                 .max_pool(3, 3, 2, 2, name='pool1')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

            (self.feed('bn2a_branch1', 
                       'bn2a_branch2c')
                 .add(name='res2a')
                 .relu(name='res2a_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

            (self.feed('res2a_relu', 
                       'bn2b_branch2c')
                 .add(name='res2b')
                 .relu(name='res2b_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

            (self.feed('res2b_relu', 
                       'bn2c_branch2c')
                 .add(name='res2c')
                 .relu(name='res2c_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

            (self.feed('res2c_relu')
                 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

            (self.feed('bn3a_branch1', 
                       'bn3a_branch2c')
                 .add(name='res3a')
                 .relu(name='res3a_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

            (self.feed('res3a_relu', 
                       'bn3b1_branch2c')
                 .add(name='res3b1')
                 .relu(name='res3b1_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

            (self.feed('res3b1_relu', 
                       'bn3b2_branch2c')
                 .add(name='res3b2')
                 .relu(name='res3b2_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

            (self.feed('res3b2_relu', 
                       'bn3b3_branch2c')
                 .add(name='res3b3')
                 .relu(name='res3b3_relu')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

            (self.feed('res3b3_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

            (self.feed('bn4a_branch1', 
                       'bn4a_branch2c')
                 .add(name='res4a')
                 .relu(name='res4a_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

            (self.feed('res4a_relu', 
                       'bn4b1_branch2c')
                 .add(name='res4b1')
                 .relu(name='res4b1_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

            (self.feed('res4b1_relu', 
                       'bn4b2_branch2c')
                 .add(name='res4b2')
                 .relu(name='res4b2_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

            (self.feed('res4b2_relu', 
                       'bn4b3_branch2c')
                 .add(name='res4b3')
                 .relu(name='res4b3_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

            (self.feed('res4b3_relu', 
                       'bn4b4_branch2c')
                 .add(name='res4b4')
                 .relu(name='res4b4_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

            (self.feed('res4b4_relu', 
                       'bn4b5_branch2c')
                 .add(name='res4b5')
                 .relu(name='res4b5_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

            (self.feed('res4b5_relu', 
                       'bn4b6_branch2c')
                 .add(name='res4b6')
                 .relu(name='res4b6_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

            (self.feed('res4b6_relu', 
                       'bn4b7_branch2c')
                 .add(name='res4b7')
                 .relu(name='res4b7_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

            (self.feed('res4b7_relu', 
                       'bn4b8_branch2c')
                 .add(name='res4b8')
                 .relu(name='res4b8_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

            (self.feed('res4b8_relu', 
                       'bn4b9_branch2c')
                 .add(name='res4b9')
                 .relu(name='res4b9_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

            (self.feed('res4b9_relu', 
                       'bn4b10_branch2c')
                 .add(name='res4b10')
                 .relu(name='res4b10_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

            (self.feed('res4b10_relu', 
                       'bn4b11_branch2c')
                 .add(name='res4b11')
                 .relu(name='res4b11_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

            (self.feed('res4b11_relu', 
                       'bn4b12_branch2c')
                 .add(name='res4b12')
                 .relu(name='res4b12_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

            (self.feed('res4b12_relu', 
                       'bn4b13_branch2c')
                 .add(name='res4b13')
                 .relu(name='res4b13_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

            (self.feed('res4b13_relu', 
                       'bn4b14_branch2c')
                 .add(name='res4b14')
                 .relu(name='res4b14_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

            (self.feed('res4b14_relu', 
                       'bn4b15_branch2c')
                 .add(name='res4b15')
                 .relu(name='res4b15_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

            (self.feed('res4b15_relu', 
                       'bn4b16_branch2c')
                 .add(name='res4b16')
                 .relu(name='res4b16_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

            (self.feed('res4b16_relu', 
                       'bn4b17_branch2c')
                 .add(name='res4b17')
                 .relu(name='res4b17_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

            (self.feed('res4b17_relu', 
                       'bn4b18_branch2c')
                 .add(name='res4b18')
                 .relu(name='res4b18_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

            (self.feed('res4b18_relu', 
                       'bn4b19_branch2c')
                 .add(name='res4b19')
                 .relu(name='res4b19_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

            (self.feed('res4b19_relu', 
                       'bn4b20_branch2c')
                 .add(name='res4b20')
                 .relu(name='res4b20_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

            (self.feed('res4b20_relu', 
                       'bn4b21_branch2c')
                 .add(name='res4b21')
                 .relu(name='res4b21_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
                 .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

            (self.feed('res4b21_relu', 
                       'bn4b22_branch2c')
                 .add(name='res4b22')
                 .relu(name='res4b22_relu')
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

            (self.feed('res4b22_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
                 .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

            (self.feed('bn5a_branch1', 
                       'bn5a_branch2c')
                 .add(name='res5a')
                 .relu(name='res5a_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
                 .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

            (self.feed('res5a_relu', 
                       'bn5b_branch2c')
                 .add(name='res5b')
                 .relu(name='res5b_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
                 .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
                 .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

            (self.feed('res5b_relu', 
                       'bn5c_branch2c')
                 .add(name='res5c')
                 .relu(name='res5c_relu')
                 .atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

            (self.feed('res5c_relu')
                 .atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

            (self.feed('res5c_relu')
                 .atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

            (self.feed('res5c_relu')
                 .atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

            (self.feed('fc1_voc12_c0', 
                       'fc1_voc12_c1', 
                       'fc1_voc12_c2', 
                       'fc1_voc12_c3')
                 .add(name='fc1_voc12'))