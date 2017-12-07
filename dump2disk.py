from PIL import Image
import numpy as np
import os
import scipy.misc
from scipy.misc import imsave
import imageio
import numpy as np
import fcn_utils
from utils import *

#
# i1 / i2 : image 1 and 2
# d1_e / d1_g : depth 1 (estimated and ground truth)
# f12_e / f12_g : flow from 1 to 2 (estimated and ground truth)
# fi1_e : image 1 as interpolated from frame 2 using optical flow
# f21_e : flow from 2 to 1 (estimated only)
# fi2_e : image 2 as interpolated from frame 1 using optical flow
# o12_e / o12_g : flow from frame 1 to 2 derived from egomotion (estimated and ground truth)
# oi1_e : image 1 as interpolated from frame 2 using odometry-based flow
# o21_e / o21_g : flow from frame 2 to 1 derived from egomotion (estimated and ground truth)
# oi2_e : image 2 as interpolated from frame 1 using odometry-based flow
# m1_e / m1_g : moving mask on image 1 (estimated and ground truth)
# s1_e / s1_g : semantic segmentations on image 1 (estimated and ground truth) NOT USED
# segout : semantic segmentations from the resnet
#

def dump2disk(vis_dir, step, results):
    vislst = {}

    ################
    assert 'img' in results
    i1_g, i2_g, v1, v2 = results['img']

    i1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i1_g.png")
    scipy.misc.imsave(i1_g_path, i1_g[0,:,:,:])
    vislst['i1'] = i1_g[0,:,:,:]

    i2_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i2_g.png")
    scipy.misc.imsave(i2_g_path, i2_g[0,:,:,:])
    vislst['i2'] = i2_g[0,:,:,:]

    i1i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_i1i2.gif")
    i1i2_images = [i1_g[0,:,:,:],i2_g[0,:,:,:]]
    imageio.mimsave(i1i2_path, i1i2_images, 'GIF')
        
    v1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_v1.png")
    scipy.misc.imsave(v1_path, v1[0,:,:,:])
    vislst['v1'] = v1[0,:,:,:]

    v2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_v2.png")
    scipy.misc.imsave(v2_path, v2[0,:,:,:])
    vislst['v2'] = v2[0,:,:,:]
    
    ################
    if 'depth' in results:
        d1_e, d1_m, d1_g = results['depth']

        d1_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_d1_e.png")
        scipy.misc.imsave(d1_e_path, d1_e[0,:,:,:])
        vislst['d1_e'] = d1_e[0,:,:,:]

        d1_m_path = os.path.join(vis_dir,'{:08}'.format(step) + "_d1_m.png")
        scipy.misc.imsave(d1_m_path, d1_m[0,:,:,:])
        vislst['d1_m'] = d1_m[0,:,:,:]

        d1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_d1_g.png")
        scipy.misc.imsave(d1_g_path, d1_g[0,:,:,:])
        vislst['d1_g'] = d1_g[0,:,:,:]

    ################
    if 'mask' in results:
        m1_e, m1_m, m1_g = results['mask']

        m1_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_m1_e.png")
        scipy.misc.imsave(m1_e_path, m1_e[0,:,:,:])
        vislst['m1_e'] = m1_e[0,:,:,:]

        m1_m_path = os.path.join(vis_dir,'{:08}'.format(step) + "_m1_m.png")
        scipy.misc.imsave(m1_m_path, m1_m[0,:,:,:])
        vislst['m1_m'] = m1_m[0,:,:,:]

        m1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_m1_g.png")
        scipy.misc.imsave(m1_g_path, m1_g[0,:,:,:])
        vislst['m1_g'] = m1_g[0,:,:,:]

    ################
    if 'flow' in results:
        f12_e, f12_g, fi1_e = results['flow']
        
        f12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f12_e.png")
        scipy.misc.imsave(f12_e_path, f12_e[0,:,:,:])
        vislst['f12_e'] = f12_e[0,:,:,:]

        f12_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f12_g.png")
        scipy.misc.imsave(f12_g_path, f12_g[0,:,:,:])
        vislst['f12_g'] = f12_g[0,:,:,:]

        fi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_fi1i1.gif")
        fi1i1_images = [i1_g[0,:,:,:],fi1_e[0,:,:,:]]
        imageio.mimsave(fi1i1_path, fi1i1_images)

        if 'bckflow' in results:
            f21_e, fi2_e = results['bckflow']

            f21_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f21_e.png")
            scipy.misc.imsave(f21_e_path, f21_e[0,:,:,:])
            vislst['f21_e'] = f21_e[0,:,:,:]

            fi2i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_fi2i2.gif")
            fi2i2_images = [i2_g[0,:,:,:],fi2_e[0,:,:,:]]
            imageio.mimsave(fi2i2_path, fi2i2_images)

    ################
    if 'odo2' in results:
        o12_e, o12_g, oi1_e, oi1_g = results['odo2']

        o12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o12_e.png")
        scipy.misc.imsave(o12_e_path, o12_e[0,:,:,:])
        vislst['o12_e'] = o12_e[0,:,:,:]

        o12_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o12_g.png")
        scipy.misc.imsave(o12_g_path, o12_g[0,:,:,:])
        vislst['o12_g'] = o12_g[0,:,:,:]

        oi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_oi1i1.gif")
        oi1i1_images = [i1_g[0,:,:,:],oi1_e[0,:,:,:]]
        imageio.mimsave(oi1i1_path, oi1i1_images)

        oi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_oi1i1_g.gif")
        oi1i1_images = [i1_g[0,:,:,:],oi1_g[0,:,:,:]]
        imageio.mimsave(oi1i1_path, oi1i1_images)
        
        if 'bckodo2' in results:
            o21_e, o21_g, oi2_e = results['bckodo2']

            o21_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o21_e.png")
            scipy.misc.imsave(o21_e_path, o21_e[0,:,:,:])
            vislst['o21_e'] = o21_e[0,:,:,:]

            o21_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_o21_g.png")
            scipy.misc.imsave(o21_g_path, o21_g[0,:,:,:])
            vislst['o21_g'] = o21_g[0,:,:,:]

            oi2i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_oi2i2.gif")
            oi2i2_images = [i2_g[0,:,:,:],oi2_e[0,:,:,:]]
            imageio.mimsave(oi2i2_path, oi2i2_images)


    ################
    if 'pose' in results:
        p12_e, p12_e2, pi1_e, pi12_e2 = results['pose']

        p12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_p12_e.png")
        scipy.misc.imsave(p12_e_path, p12_e[0,:,:,:])
        vislst['p12_e'] = p12_e[0,:,:,:]

        p12_e2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_p12_e2.png")
        scipy.misc.imsave(p12_e2_path, p12_e2[0,:,:,:])
        vislst['p12_e2'] = p12_e2[0,:,:,:]

        pi1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_pi1i1.gif")
        pi1i1_images = [i1_g[0,:,:,:],pi1_e[0,:,:,:]]
        imageio.mimsave(pi1i1_path, pi1i1_images)
        
        if 'bckpose' in results:
            p21_e, p21_e2, pi2_e, pi2_e2 = results['bckpose']

            p2i1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_pi1i1_2.gif")
            p2i1i1_images = [i1_g[0,:,:,:],pi1_e2[0,:,:,:]]
            imageio.mimsave(p2i1i1_path, p2i1i1_images)

    if 'com' in results:
        c12_e, ci1_e, ti1_e, ti2_e = results['com']

        c12_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_c12_e.png")
        scipy.misc.imsave(c12_e_path, c12_e[0,:,:,:])
        vislst['c12_e'] = c12_e[0,:,:,:]
        ci1i1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_ci1i1.gif")
        ci1i1_images = [i1_g[0,:,:,:],ci1_e[0,:,:,:]]
        imageio.mimsave(ci1i1_path, ci1i1_images)

        ti1i2_path = os.path.join(vis_dir,'{:08}'.format(step) + "_ti1i2.gif")
        ti1i2_images = [ti1_e[0,:,:,:],ti2_e[0,:,:,:]]
        imageio.mimsave(ti1i2_path, ti1i2_images)

    if 'objdepth' in results:
        r1_g, bx1_g, d1_g = results['objdepth']

        def box2img(boxes):
            canvas = np.zeros((hyp.h, hyp.w))
            for (L,T,R,B) in boxes:
                #print L,T,R,B
                canvas[T:B,L:R] = 1.0
            return canvas

        r1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_r1_g.png")
        bx1_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_bx1_g.png")
        r1_g[r1_g < 0.0] = 0.0
        r1_g[0,:,:,0] /= (np.max(r1_g[0,:,:,0])+1E-12)
        r1_g[0,:,:,0] *= 3

        scipy.misc.imsave(r1_g_path, r1_g[0,:,:,:])
        scipy.misc.imsave(bx1_g_path, box2img(bx1_g[0]))
        vislst['r1_g'] = r1_g[0,:,:,:]
        
        depth_diff = np.abs(d1_g[:,:,:,0]-r1_g[:,:,:,0])
        depth_diff *= r1_g[:,:,:,1] #valid mask
        dd_path = os.path.join(vis_dir,'{:08}'.format(step) + "_dd_g.png")
        scipy.misc.imsave(dd_path, depth_diff[0])
        vislst['dd_g'] = depth_diff[0]

    if 'seg' in results:
        seg_probs, s1_g, s1_e, seg_loss = results['seg']
        if s1_e is not None:
            seg_path_e = os.path.join(vis_dir,'{:08}'.format(step) + "_s1_e.png")
            seg_path_g = os.path.join(vis_dir,'{:08}'.format(step) + "_s1_g.png")

            rgb_seg_e = fcn_utils.color_image(s1_e[0,:,:], hyp.num_seg_classes)
            rgb_seg_g = fcn_utils.color_image(s1_g[0,:,:,0], hyp.num_seg_classes)

            scipy.misc.imsave(seg_path_e, rgb_seg_e)
            scipy.misc.imsave(seg_path_g, rgb_seg_g)
            
    if 'normals' in results:
        sn_e, sn_g, sm_e = results['normals']

        # n_classes = 14
        # #'building, car, guardrail, misc, pole, road, sky, terrain, trafficlight, trafficsign, tree, truck, van, vegetation'
        # label_colours = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
        #                  (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
        #                  (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
        #                  (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
        #                  (1.0, 0.4980392156862745, 0.0),
        #                  (1.0, 1.0, 0.2),
        #                  (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
        #                  (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
        #                  (0.6, 0.6, 0.6),
        #                  (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
        #                  (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
        #                  (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
        #                  (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
        #                  (1.0, 0.4980392156862745, 0.0)]
        # for i, tup in enumerate(label_colours):
        #     label_colours[i] = tuple(map(lambda x: int(x*255), tup))
        # s1_g_z = np.squeeze(decode_labels(s1_g, label_colours))
        # s1_path = os.path.join(vis_dir,'{:08}'.format(step) + "_s1_g.png")
        # scipy.misc.imsave(s1_path, s1_g_z)
        # vislst['s1_g'] = s1_g_z

        sm_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_sm_e.png")
        scipy.misc.imsave(sm_e_path, sm_e[0,:,:,:])
        vislst['sm_e'] = sm_e[0,:,:,:]
        
        sn_e_path = os.path.join(vis_dir,'{:08}'.format(step) + "_sn_e.png")
        scipy.misc.imsave(sn_e_path, sn_e[0,:,:,:])
        vislst['sn_e'] = sn_e[0,:,:,:]
    
        sn_g_path = os.path.join(vis_dir,'{:08}'.format(step) + "_sn_g.png")
        scipy.misc.imsave(sn_g_path, sn_g[0,:,:,:])
        vislst['sn_g'] = sn_g[0,:,:,:]
    
    do_combined = True
    if not do_combined:
        return

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

#fix this later
'''
    # these seg guys probably need work
    if s1_e is not None:
        seg_path_e = os.path.join(vis_dir,'{:08}'.format(step) + "_f1_seg_e.png")
        seg_path_g = os.path.join(vis_dir,'{:08}'.format(step) + "_f1_seg_g.png")
        rgb_seg_e = fcn_utils.color_image(s1_e[0,:,:])
        rgb_seg_g = fcn_utils.color_image(s1_g[0,:,:,0])
        scipy.misc.imsave(seg_path_e, rgb_seg_e)
        print np.shape(s1_g)
        print np.shape(rgb_seg_g)
        scipy.misc.imsave(seg_path_g, rgb_seg_g)
    if segout is not None:
        n_classes = 21
        label_colours = [(0,0,0)
                         # 0=background
                         ,(128, 64,128),(244, 35,232),( 70, 70, 70),(102,102,156),(190,153,153)
                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                         ,(153,153,153),(250,170, 30),(220,220,  0),(107,142, 35), (152,251,152)
                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                         ,( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70)
                         # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                         ,(  0, 60,100),(  0, 80,100),(  0,  0,230),(119, 11, 32),(255,225,255)]
                        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        seg_img = decode_labels(segout, label_colours)
        seg_img = np.squeeze(seg_img)
        seg_path = os.path.join(vis_dir,'{:08}'.format(step) + "_f1_resnet.png")
        scipy.misc.imsave(seg_path, seg_img)

'''
