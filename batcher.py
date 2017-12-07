import tensorflow as tf
from utils import *
import os
import hyperparams_gan as hyp

def kitti2_batch(dataset,bs,crop_h,crop_w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)
    
    h,w,have_rt,i1,i2,p1,p2,k1 = read_and_decode_kitti2(queue)
    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(2,[i1,i2],
                       name="allCat")
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")

    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        p1,p2,
                                        k1,
                                        h,w,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                p1,p2,
                                k1,
                                h,w,
                                off_h,off_w],
                               batch_size=bs)
    return batch

def kitti_batch(dataset,bs,crop_h,crop_w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)
    
    h,w,have_rt,i1,i2,rt12,k1 = read_and_decode_kitti_odometry(queue)
    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(2,[i1,i2],
                       name="allCat")
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")

    # fy = k1[1,1]
    # fx = k1[0,0]
    # y0 = k1[1,2]
    # x0 = k1[0,2]
    # # grab a batch
    # if shuffle:
    #     # this will shuffle BEYOND JUST THE BATCH!
    #     batch = tf.train.shuffle_batch([i1,i2,
    #                                     rt12,
    #                                     fy,fx,y0,x0,
    #                                     h,w,
    #                                     off_h,off_w],
    #                                    batch_size=bs,
    #                                    min_after_dequeue=100,
    #                                    capacity=100+3*bs,
    #                                    num_threads=2)
    # else:
    #     batch = tf.train.batch([i1,i2,
    #                             rt12,
    #                             fy,fx,y0,x0,
    #                             h,w,
    #                             off_h,off_w],
    #                            batch_size=bs)
    # return batch
    
    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        rt12,
                                        k1,
                                        h,w,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=hyp.queue_capacity-3*hyp.bs,
                                       capacity= hyp.queue_capacity,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                rt12,
                                k1,
                                h,w,
                                off_h,off_w],
                               batch_size=bs,
                               capacity = hyp.queue_capacity)
    return batch

def vkitti_batch(dataset,bs,crop_h,crop_w,shuffle=True,dotrim=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h,w,i1,i2,s1,s2,d1,d2,f12,f23,v1,v2,p1,p2,m1,m2,
     o1c, o1i, o1b, o1p, o1o,
     o2c, o2i, o2b, o2p, o2o,
     nc1,nc2,car1,car2) = read_and_decode_vkitti_w_everything4(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    s1 = tf.cast(s1,tf.float32)
    s2 = tf.cast(s2,tf.float32)
    car1 = tf.cast(car1,tf.float32)
    car2 = tf.cast(car2,tf.float32)

    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = tf.cast(v1,tf.float32)
    v2 = tf.cast(v2,tf.float32)
    m1 = tf.cast(m1,tf.float32) * 1./255 # for some reason these are stored in [0,255]
    m2 = tf.cast(m2,tf.float32) * 1./255

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(2,[i1,i2,
                          s1,s2,
                          d1,d2,
                          f12,f23,
                          v1,v2,
                          m1,m2,
                          car1, car2],
                       name="allCat")

    # image tensors need to be cropped. we'll do them all at once.

    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    s1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="s1")
    s2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="s2")
    d1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="d2")
    f12 = tf.slice(allCat_crop, [0,0,10], [-1,-1,2], name="f12")
    f23 = tf.slice(allCat_crop, [0,0,12], [-1,-1,2], name="f23")
    v1 = tf.slice(allCat_crop, [0,0,14], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,15], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,16], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,17], [-1,-1,1], name="m2")
    car1 = tf.slice(allCat_crop, [0,0,18],[-1,-1,1], name='car1')
    car2 = tf.slice(allCat_crop, [0,0,19],[-1,-1,1], name='car2')
    # cast the seg labels back to int
    s1 = tf.cast(s1, tf.int32)
    s2 = tf.cast(s2, tf.int32)
    car1 = tf.cast(car1, tf.int32)
    car2 = tf.cast(car2, tf.int32)

    #first scale the bboxes
    o1b = tf.cast(tf.cast(o1b, tf.float32)*hyp.scale, tf.int64)
    o2b = tf.cast(tf.cast(o2b, tf.float32)*hyp.scale, tf.int64)
    
    #we need to shift the object bounding boxes over a bit
    #this function is in utils
    o1b = offsetbbox(o1b, off_h, off_w, crop_h, crop_w, dotrim)
    o2b = offsetbbox(o2b, off_h, off_w, crop_h, crop_w, dotrim)

    

    # grab a batch
    if shuffle and (hyp.dataset_name != 'VKITTI'):
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        s1,s2,
                                        d1,d2,
                                        f12,f23,
                                        v1,v2,
                                        p1,p2,
                                        m1,m2,
                                        off_h,off_w,
                                        o1c,o1i,o1b,o1p,o1o,
                                        o2c,o2i,o2b,o2p,o2o,
                                        nc1,nc2,car1,car2],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                s1,s2,
                                d1,d2,
                                f12,f23,
                                v1,v2,
                                p1,p2,
                                m1,m2,
                                off_h,off_w,
                                o1c,o1i,o1b,o1p,o1o,
                                o2c,o2i,o2b,o2p,o2o,
                                nc1,nc2,car1,car2],
                               batch_size=bs,
                               dynamic_pad = True)
    return batch

def toy_batch(dataset,bs,h,w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    
    nRecords = len(records)
    print 'found %d records' % nRecords
    queue = tf.train.string_input_producer(records, shuffle=shuffle)
    
    height,width,i1,i2,d1,d2,v1,v2,m1,m2 = read_and_decode_toy(queue)

    # some tensors need to be cast to float
    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = 1-tf.cast(v1,tf.float32) # this has ones at INVALID pixels
    v2 = 1-tf.cast(v2,tf.float32)
    m1 = tf.cast(m1,tf.float32) * 1./255 # this is in [0,255]
    m2 = tf.cast(m2,tf.float32) * 1./255

    # we need to clean up the valid mask a bit
    d1_ = tf.expand_dims(d1,0)
    d2_ = tf.expand_dims(d2,0)
    v1_ = tf.expand_dims(v1,0)
    v2_ = tf.expand_dims(v2,0)
    # high pass on depth to find more invalids
    blur_kernel = tf.transpose(tf.constant([[[[1./16,1./8,1./16],
                                              [1./8,1./4,1./8],
                                              [1./16,1./8,1./16]]]],
                                           dtype=tf.float32),perm=[3,2,1,0])
    blurred_d1 = tf.nn.conv2d(d1_, blur_kernel, strides=[1,1,1,1], padding="SAME")
    blurred_d2 = tf.nn.conv2d(d2_, blur_kernel, strides=[1,1,1,1], padding="SAME")
    sharp_d1 = d1_-blurred_d1
    sharp_d2 = d2_-blurred_d2
    zero_kernel = tf.zeros([3,3,1])
    # also erode valid
    v1_ = tf.nn.erosion2d(
        tf.cast(tf.less(tf.abs(sharp_d1),0.01*tf.ones_like(sharp_d1)),tf.float32)*v1_,
        zero_kernel,[1,1,1,1],[1,1,1,1], "SAME")
    v2_ = tf.nn.erosion2d(
        tf.cast(tf.less(tf.abs(sharp_d2),0.01*tf.ones_like(sharp_d2)),tf.float32)*v2_,
        zero_kernel,[1,1,1,1],[1,1,1,1], "SAME")
    # d1_ = tf.squeeze(d1_,0)
    # d2_ = tf.squeeze(d2_,0)
    v1_ = tf.squeeze(v1_,0)
    v2_ = tf.squeeze(v2_,0)

    amount = 20
    if hyp.do_crop_aug:
        # we'll crop the first tensors slightly, then crop the second tensors, then continue
        allCat = tf.concat(2,[i1,
                              d1,
                              v1_,
                              m1],
                           name="allCat")
        allCat_crop, off_h, off_w = near_topleft_crop(allCat,height-amount,width-amount,height,width,amount)
        i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
        d1 = tf.slice(allCat_crop, [0,0,3], [-1,-1,1], name="d1")
        v1_ = tf.slice(allCat_crop, [0,0,4], [-1,-1,1], name="v1")
        m1 = tf.slice(allCat_crop, [0,0,5], [-1,-1,1], name="m1")

        allCat = tf.concat(2,[i2,
                              d2,
                              v2_,
                              m2],
                           name="allCat")
        allCat_crop, off_h, off_w = near_topleft_crop(allCat,height-amount,width-amount,height,width,amount)
        i2 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i2")
        d2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,1], name="d2")
        v2_ = tf.slice(allCat_crop, [0,0,4], [-1,-1,1], name="v2")
        m2 = tf.slice(allCat_crop, [0,0,5], [-1,-1,1], name="m2")

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(2,[i1,i2,
                          d1,d2,
                          v1_,v2_,
                          m1,m2],
                       name="allCat")
    print_shape(allCat)
    if hyp.do_crop_aug:
        allCat_crop, off_h, off_w = near_topleft_crop(allCat,h,w,height-amount,width-amount,10)
    else:
        allCat_crop, off_h, off_w = near_topleft_crop(allCat,h,w,height,width,10)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    v1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,10], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,11], [-1,-1,1], name="m2")

    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        d1,d2,
                                        v1,v2,
                                        m1,m2,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                d1,d2,
                                v1,v2,
                                m1,m2,
                                off_h,off_w],
                               batch_size=bs)


    return batch


def ycb_batch(dataset,bs,crop_h,crop_w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    h,w,i1,i2,d1,d2,v1,v2,m1,m2,p1,p2,k1,k2 = read_and_decode_ycb(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = 1-tf.cast(v1,tf.float32)
    v2 = 1-tf.cast(v2,tf.float32)
    m1 = 1-(tf.cast(m1,tf.float32) * 1./255)
    m2 = 1-(tf.cast(m2,tf.float32) * 1./255)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(2,[i1,i2,
                          d1,d2,
                          v1,v2,
                          m1,m2],
                       name="allCat")
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    v1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,10], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,11], [-1,-1,1], name="m2")

    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        d1,d2,
                                        v1,v2,
                                        m1,m2,
                                        p1,p2,
                                        k1,k2,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                d1,d2,
                                v1,v2,
                                m1,m2,
                                p1,p2,
                                k1,k2,
                                off_h,off_w],
                               batch_size=bs)
    return batch

