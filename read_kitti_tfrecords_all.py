from PIL import Image
import scipy.misc
import os
import imageio
import numpy as np
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
import glob, os
from utils import *

bs=1
scale=0.5
# scale=1.0
# HEIGHT = int(1024*scale)
# WIDTH = int(1280*scale)
HEIGHT = 180
WIDTH = 600

with open("datasets/kitti_odometry/one_of_each.txt") as f:
    content = f.readlines()
records = ['datasets/kitti_odometry/records/' + line.strip() for line in content]

# for i in range(0,5):
#     print records[i]

nRecords = len(records)
print 'found %d records' % nRecords
queue = tf.train.string_input_producer(records, shuffle=False)

h,w,have_rt,i1,i2,rt12,k1 = read_and_decode_kitti_odometry(queue)

# some tensors need to be cast to float
i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5

# image tensors need to be cropped. we'll do them all at once.
allCat = tf.concat(2,[i1,i2],
                   name="allCat")
print_shape(allCat)
allCat_crop, off_h, off_w = random_crop(allCat,HEIGHT,WIDTH,h,w)
print_shape(allCat_crop)
i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")

# grab a batch
batch = tf.train.batch([i1,i2,
                        rt12,
                        k1,
                        h,w,
                        off_h,off_w],
                       batch_size=bs)
i1,i2,rt12,k1,h,w,off_h,off_w = batch

fx,fy,x0,y0 = split_intrinsics(k1)
fx=fx*scale
fy=fy*scale
x0=x0*scale
y0=y0*scale

# prep some visualizations of the trickier pieces of data
i1_z = back2image(i1)
i2_z = back2image(i2)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(nRecords):
        print "%d/%d; %s" % (step, nRecords, records[step])
        i1_,i2_,fx_,fy_,x0_,y0_,h_,w_ = sess.run([i1_z,i2_z,fx,fy,x0,y0,h,w])
        # raw_input()
    coord.request_stop()
    coord.join(threads)
