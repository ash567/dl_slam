import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import hyperparams_gan as hyp

def save(saver, sess, checkpoint_dir, step):
    model_name = "minuet.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)
    print("Saved a checkpoint: %s/%s-%d" % (checkpoint_dir, model_name, step))

def load(saver, sess, model):
    print("reading full checkpoint...")

    print len(tf.trainable_variables())
    for v in tf.trainable_variables():
        print 'name = {}'.format(v.value())

    model_dir = os.path.join("checkpoint", model)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    start_iter = 0
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("...found %s " % ckpt.model_checkpoint_path)
        start_iter = int(ckpt_name[len("minuet-model")+1:])
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
    else:
        print("...ain't no full checkpoint here!")

    if not hyp.do_resnet:
        return start_iter

    resnetckpt = 'resnet/snapshots/baseVkitti.ckpt'
    resnet_dict_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'inference/resnet/')
    resnet_dict_keys = [T.name for T in resnet_dict_values]
    resnet_dict_keys = [name.replace('inference/resnet/','netShare/').replace(':0','') for name in resnet_dict_keys]
    resnet_dict = {k:v for (k,v) in zip(resnet_dict_keys, resnet_dict_values)}
    resnet_saver = tf.train.Saver(resnet_dict)
    if os.path.isfile(resnetckpt+'.meta'):
        print 'restoring resnet model'
        resnet_saver.restore(sess, resnetckpt)
    else:
        print 'resnet weights not found'

    return start_iter

def load_part(sess, model, part):
    print "reading %s checkpoint..." % part
    model_dir = os.path.join("checkpoint", model)
    print model_dir
    ckpt = tf.train.get_checkpoint_state(model_dir)
    start_iter = 0
    if ckpt and ckpt.model_checkpoint_path and part is not "seg":
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("...found %s " % ckpt.model_checkpoint_path)
        start_iter = int(ckpt_name[len("minuet-model")+1:])
        model_vars = slim.get_model_variables()
        scope = "inference/%s/%sNet" % (part, part.title())
        print "loading %s part" % scope
        my_vars = slim.get_variables_to_restore(include=[scope])
        vars_to_restore = set(model_vars).intersection(my_vars)
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(sess, os.path.join(model_dir, ckpt_name))
    elif ckpt and ckpt.model_checkpoint_path and part is "seg":
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("...found %s " % ckpt.model_checkpoint_path)
        start_iter = int(ckpt_name[len("minuet-model")+1:])
        model_vars = slim.get_model_variables()
        restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not hyp.not_restore_last]
        scope = "inference/seg/model/Resnet"
        my_vars = slim.get_variables_to_restore(include=[scope])
        vars_to_restore = set(restore_var).intersection(my_vars)
        
        # scope = "inference/seg/model/Resnet"
        print "loading %s part" % scope
        # my_vars = [v for v in my_vars if ('moving_mean' not in v.name and 'moving_variance' not in v.name and 'beta' not in v.name and 'gamma' not in v.name)]
        # vars_to_restore = set(model_vars).intersection(my_vars)
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(sess, os.path.join(model_dir, ckpt_name))
    else:
        print "...ain't no %s checkpoints here!" % part
    return start_iter

def load_rcnn(sess):
    model_vars = slim.get_model_variables()
    scope = "inference/rcnn"
    my_vars = slim.get_variables_to_restore(include=[scope])

    __d = 'tf_faster_rcnn/coco_2014_train+coco_2014_valminusminival/res101_faster_rcnn_iter_1190000.ckpt'
    __f = lambda t: t.name.replace('inference/rcnn/', '').replace(':0', '')
    __g = lambda t: sum((1 for val in ['mask_net', 'size_pred', 'pose_pred'] 
                         if val in t.name)) == 0
    rd = {__f(x): x for x in my_vars if __g(x)}
    rd2 = {k:v for k,v in rd.items() if (k.find("Adam")==-1)}

    # print rd2.keys()
    # print '--- and that\'s all the keys! ---'

    restorer = tf.train.Saver(rd2)
    restorer.restore(sess, __d)
    
