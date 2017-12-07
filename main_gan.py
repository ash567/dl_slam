import os
import cv2
from model_gan_modified import Minuet
import tensorflow as tf
import hyperparams_gan as hyp

def main(_):
    checkpoint_dir_ = os.path.join("checkpoint", hyp.name)
    vis_dir_ = os.path.join("vis", hyp.name)
    log_dir_ = os.path.join("log", hyp.name)
    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(vis_dir_):
        os.makedirs(vis_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)
    if not os.path.exists('poses'):
        os.makedirs('poses')
        
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    with tf.Session(config=c) as sess:
        minuet = Minuet(sess,
                        checkpoint_dir=checkpoint_dir_,
                        vis_dir=vis_dir_, 
                        log_dir=log_dir_
        )
        minuet.go()

if __name__ == '__main__':
    tf.app.run()
    
