from PIL import Image
import scipy.misc
import os
import numpy as np
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
import glob, os
from utils import *

# set these to anything, but make sure they're below the maxes
scale=0.4
HEIGHT = int(375*scale)
WIDTH = int(1242*scale)

# with open("datasets/vkitti/lists/trainval.txt") as f:
with open("datasets/vkitti/lists/some.txt") as f:
    content = f.readlines()
records = ['datasets/vkitti/singles_w_everything4/' + line.strip() for line in content]
    
nRecords = len(records)
print 'found %d records' % nRecords
queue = tf.train.string_input_producer(records, shuffle=False)

(h,w,i1,i2,s1,s2,d1,d2,f1,f2,v1,v2,p1,p2,m1,m2,
 o1c,o1i,o1b,o1p,o1o,
 o2c,o2i,o2b,o2p,o2o,
 nCars, car1, car2) = read_and_decode_vkitti_w_everything4(queue)

print_shape(i1)
print_shape(i2)
print_shape(s1)
print_shape(s2)
print_shape(d1)
print_shape(d2)
print_shape(f1)
print_shape(f2)
print_shape(v1)
print_shape(v2)
print_shape(p1)
print_shape(p2)
print_shape(m1)
print_shape(m2)
print_shape(o1c)
print_shape(o1i)
print_shape(o1b)
print_shape(o1p)
print_shape(o1o)
print_shape(o2c)
print_shape(o2i)
print_shape(o2b)
print_shape(o2p)
print_shape(o2o)
print_shape(nCars)
print_shape(car1)
print_shape(car2)

# some tensors need to be cast to float
i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
s1 = tf.cast(s1,tf.float32)
s2 = tf.cast(s2,tf.float32)
d1 = tf.cast(d1,tf.float32)
d2 = tf.cast(d2,tf.float32)
v1 = tf.cast(v1,tf.float32)
v2 = tf.cast(v2,tf.float32)
m1 = tf.cast(m1,tf.float32)
m2 = tf.cast(m2,tf.float32)
c1 = tf.cast(car1,tf.float32)
c2 = tf.cast(car2,tf.float32)

# image tensors need to be cropped. we'll do them all at once.
allCat = tf.concat(2,[i1,i2,
                      s1,s2,
                      d1,d2,
                      f1,f2,
                      v1,v2,
                      m1,m2,
                      c1,c2],
                   name="allCat")
print_shape(allCat)
allCat_crop, off_h, off_w = random_crop(allCat,HEIGHT,WIDTH,h,w)
print_shape(allCat_crop)
i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
s1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="s1")
s2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="s2")
d1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="d1")
d2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="d2")
f1 = tf.slice(allCat_crop, [0,0,10], [-1,-1,2], name="f1")
f2 = tf.slice(allCat_crop, [0,0,12], [-1,-1,2], name="f2")
v1 = tf.slice(allCat_crop, [0,0,14], [-1,-1,1], name="v1")
v2 = tf.slice(allCat_crop, [0,0,15], [-1,-1,1], name="v2")
m1 = tf.slice(allCat_crop, [0,0,16], [-1,-1,1], name="m1")
m2 = tf.slice(allCat_crop, [0,0,17], [-1,-1,1], name="m2")
c1 = tf.slice(allCat_crop, [0,0,18], [-1,-1,1], name="c1")
c2 = tf.slice(allCat_crop, [0,0,19], [-1,-1,1], name="c2")

# grab a batch
batch = tf.train.batch([i1,i2,
                        s1,s2,
                        d1,d2,
                        f1,f2,
                        v1,v2,
                        p1,p2,
                        m1,m2,
                        c1,c2,
                        off_h,off_w,],
                       batch_size=1)
i1,i2,s1,s2,d1,d2,f1,f2,v1,v2,p1,p2,m1,m2,c1,c2,off_h,off_w = batch




# prep some visualizations of the trickier pieces of data

i1_z = back2color(i1)
i2_z = back2color(i2)
s1_z = oned2color(s1)
s2_z = oned2color(s2)
d1_z = oned2color(d1)
d2_z = oned2color(d2)
f1_z = flow2color(f1)
f2_z = flow2color(f2)
v1_z = oned2color(v1)
v2_z = oned2color(v2)
m1_z = oned2color(m1)
m2_z = oned2color(m2)
# c1_z = oned2color(c1)
# c2_z = oned2color(c2)
c1_z = oned2color(tf.select(tf.greater(c1, 0), tf.ones_like(c1), tf.zeros_like(c1)))
c2_z = oned2color(tf.select(tf.greater(c2, 0), tf.ones_like(c1), tf.zeros_like(c1)))

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(nRecords):
        print "step = %d; %s" % (step, records[step])
        try:
            i1_,i2_,s1_,s2_,d1_,d2_,f1_,f2_,v1_,v2_,p1_,p2_,m1_,m2_,c1_,c2_,off_h_,off_w_ \
                = sess.run([i1_z,i2_z,s1_z,s2_z,d1_z,d2_z,f1_z,f2_z,v1_z,v2_z,p1,p2,m1_z,m2_z,c1_z,c2_z,off_h,off_w])
            Image.fromarray(np.uint8(i1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(i2_[0, :, :, :])).show()
            
            Image.fromarray(np.uint8(s1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(s2_[0, :, :, :])).show()

            # Image.fromarray(np.uint8(d1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(d2_[0, :, :, :])).show()

            # Image.fromarray(np.uint8(f1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(f2_[0, :, :, :])).show()

            # Image.fromarray(np.uint8(v1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(v2_[0, :, :, :])).show()

            # Image.fromarray(np.uint8(m1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(m2_[0, :, :, :])).show()

            Image.fromarray(np.uint8(c1_[0, :, :, :])).show()
            # Image.fromarray(np.uint8(c2_[0, :, :, :])).show()

            print p1_[0]
            print p2_[0]

            print off_h_[0]
            print off_w_[0]

            raw_input()
            
        except:
            print "howwohwoowo!!"
            raw_input()
    coord.request_stop()
    coord.join(threads)
