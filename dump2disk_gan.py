from PIL import Image
import numpy as np
import os
import scipy.misc
from scipy.misc import imsave
import imageio
import numpy as np
import fcn_utils
from utils import *
import hyperparams_gan as hyp


# Later add segmentation to the dump2disk

def dump2disk_gan(vis_dir, step, i1, i2, d1_e, d1_g, f12_e_z, f12_g_z, f21_e_z, fi1_e_z, fi2_e_z,
 o12_e_z, o12_g_z, o21_e_z, o21_g_z, oi1_e_z, oi2_e_z, m1_g_z):

    vislst = {}

    # IMAGE
    ################
    i1_g = i1
    i2_g = i2

    i1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i1_g.png")
    scipy.misc.imsave(i1_g_path, i1_g[0,:,:,:])
    vislst['i1'] = i1_g[0,:,:,:]

    i2_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i2_g.png")
    scipy.misc.imsave(i2_g_path, i2_g[0,:,:,:])
    vislst['i2'] = i2_g[0,:,:,:]

    i1i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i1i2.gif")
    i1i2_images = [i1_g[0,:,:,:],i2_g[0,:,:,:]]
    imageio.mimsave(i1i2_path, i1i2_images, 'GIF')


    # Depth
    ################

    # d1_e = d1_e
    # d1_g = 
    # Variable names are the same

    if hyp.do_depth:
        d1_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_d1_e.png")
        scipy.misc.imsave(d1_e_path, d1_e[0,:,:,:])
        vislst['d1_e'] = d1_e[0,:,:,:]

        d1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_d1_g.png")
        scipy.misc.imsave(d1_g_path, d1_g[0,:,:,:])
        vislst['d1_g'] = d1_g[0,:,:,:]

    ################
    # MASK
    if hyp.do_mask:
        m1_g = m1_g_z

        m1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_m1_g.png")
        scipy.misc.imsave(m1_g_path, m1_g[0,:,:,:])
        vislst['m1_g'] = m1_g[0,:,:,:]

    ################
    # # FLO
        
    # f12_e = f12_e_z
    # f12_g = f12_g_z
    # fi1_e = fi1_e_z
    
    # f12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f12_e.png")
    # scipy.misc.imsave(f12_e_path, f12_e[0,:,:,:])
    # vislst['f12_e'] = f12_e[0,:,:,:]

    # f12_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f12_g.png")
    # scipy.misc.imsave(f12_g_path, f12_g[0,:,:,:])
    # vislst['f12_g'] = f12_g[0,:,:,:]

    # fi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_fi1i1.gif")
    # fi1i1_images = [i1_g[0,:,:,:],fi1_e[0,:,:,:]]
    # imageio.mimsave(fi1i1_path, fi1i1_images)

    
    # f21_e = f21_e_z
    # fi2_e = fi2_e_z

    # f21_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f21_e.png")
    # scipy.misc.imsave(f21_e_path, f21_e[0,:,:,:])
    # vislst['f21_e'] = f21_e[0,:,:,:]

    # fi2i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_fi2i2.gif")
    # fi2i2_images = [i2_g[0,:,:,:],fi2_e[0,:,:,:]]
    # imageio.mimsave(fi2i2_path, fi2i2_images)

    # ODO

    if hyp.do_odo:
        o12_e = o12_e_z
        o12_g = o12_g_z
        oi1_e = oi1_e_z

        o12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o12_e.png")
        scipy.misc.imsave(o12_e_path, o12_e[0,:,:,:])
        vislst['o12_e'] = o12_e[0,:,:,:]

        o12_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o12_g.png")
        scipy.misc.imsave(o12_g_path, o12_g[0,:,:,:])
        vislst['o12_g'] = o12_g[0,:,:,:]


        oi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_oi1i1.gif")
        oi1i1_images = [i1_g[0,:,:,:],oi1_e[0,:,:,:]]
        imageio.mimsave(oi1i1_path, oi1i1_images)



        o21_e = o21_e_z
        o21_g = o21_g_z
        oi2_e = oi2_e_z


        o21_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o21_e.png")
        scipy.misc.imsave(o21_e_path, o21_e[0,:,:,:])
        vislst['o21_e'] = o21_e[0,:,:,:]

        o21_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o21_g.png")
        scipy.misc.imsave(o21_g_path, o21_g[0,:,:,:])
        vislst['o21_g'] = o21_g[0,:,:,:]

        oi2i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_oi2i2.gif")
        oi2i2_images = [i2_g[0,:,:,:],oi2_e[0,:,:,:]]
        imageio.mimsave(oi2i2_path, oi2i2_images)

    # # if 'seg' in results:
    # seg_probs, s1_g, s1_e, seg_loss = results['seg']
    # if s1_e is not None:
    #     seg_path_e = os.path.join(vis_dir,'{:08}'.format(step) + "_s1_e.png")
    #     seg_path_g = os.path.join(vis_dir,'{:08}'.format(step) + "_s1_g.png")

    #     rgb_seg_e = fcn_utils.color_image(s1_e[0,:,:], hyp.num_seg_classes)
    #     rgb_seg_g = fcn_utils.color_image(s1_g[0,:,:,0], hyp.num_seg_classes)

    #     scipy.misc.imsave(seg_path_e, rgb_seg_e)
    #     scipy.misc.imsave(seg_path_g, rgb_seg_g)
            
    
    
    order = ['i1', 'i2', 
             'v1', 'v2', 
             'd1_e', 'd1_m', 'd1_g', 
             'm1_e', 'm1_m', 'm1_g', 
             'f12_e', 'f12_g', 'f21_e',
             'o12_e', 'o12_g', 'o21_e', 'o21_g',
             's1_g', 
             'p12_e', 'p12_e2', 
             'c12_e', 'r1_g', 'dd_g', 'sn_e']

    numimgs = len(vislst.keys())
    numcols = 3
    numrows = (numimgs-1)/numcols+1 #updiv

    canvas = np.zeros((numrows*hyp.h, numcols*hyp.w, 3))
    assert numcols * numrows >= numimgs
    count = 0
    for key in order:
        if key in vislst:
            img = vislst[key].astype(np.float32)
            img -= np.min(img)
            img /= (np.max(img)+hyp.eps)
            numdims = len(np.shape(img))
            assert 2 <= numdims <= 3
            if numdims == 2:
                img = np.stack([img, img, img], axis = 2)

            #going down rows
            row = count%numrows
            col = count/numrows
            canvas[row*hyp.h:(row+1)*hyp.h, col*hyp.w:(col+1)*hyp.w,:] = img
            count += 1

    #fix this directory
    _dir = os.path.join(vis_dir, 'all')
    __dir = os.path.join(_dir, '{:08}'.format(step)+'.png')
    if not os.path.exists(_dir):
        os.mkdir(_dir)

    imsave(__dir, canvas)