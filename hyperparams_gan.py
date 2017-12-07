depth_init = ""
mask_init = ""
odo_init = ""
# flow_init = "02x128x384_1.0e-05_F4_p10.0_s1.0_classic_trainval"
flow_init = ""
flow_init = ""
pose_init = ""
total_init = ""
seg_init = ""
gan_init = ""

# flow_init = "02x128x512_1.0e-04_F4_p10.0_s1.0_backward_KITTI_train_init"
# depth_init = "02x128x512_1.0e-04_D4_1.0_l1_sky0.01_s0.0_vkitti"
# depth_init = "02x128x512_1.0e-04_D4_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_fixed2"
# odo_init = "02x128x512_1.0e-04_D4_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_fixed2"
# total_init = "02x128x512_1.0e-04_D4_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_fixed2" # this guy seemed to generalize best on vkitti
# odo_init = "01x64x64_1.0e-05_D1_0.0_inval0.00_s0.1_p0.1_obj1.0_O1_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r+a_VKITTI_val_bench_debug"

# flow_init = "01x64x64_1.0e-05_D1_0.0_inval0.00_s0.1_p0.1_obj1.0_O1_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r+a_VKITTI_val_bench_debug"
# total_init = "02x128x512_1.0e-04_D4_1.0_l1_sky0.01_s0.0_vkitti"

# depth_init = "02x512x512_1.0e-04_D4_1.0_sky0.00_s0.0_p0.0_YCB_list"
# total_init = "02x128x512_1.0e-05_D4_F4_O4_p10.00+r_VKITTI_val"
# total_init = "02x128x512_1.0e-05_D4_F4_O5_p10.00+r_fc_VKITTI_val"
# total_init = "02x128x512_1.0e-05_D4_O5_p10.00_rtd1.0_rta100.0_td1.0_ta1.0+r_fc_VKITTI_train"
# total_init = "02x128x512_1.0e-05_D4_F4_O5_p10.00+r_fc_VKITTI_val"
# total_init = "02x128x512_1.0e-05_D4_O6_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r_fc_VKITTI_val"
# total_init = "02x128x512_1.0e-05_D4_0.0_sky0.00_s0.0_p1.0_O6_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r_fc_VKITTI_val_totalunsup"
# total_init = "02x128x512_1.0e-05_D4_O5_p10.00+r_fc_VKITTI_val_ft"

dataset_name = 'VKITTI'
# dataset_name = 'KITTI'
# dataset_name = 'KITTI2'
# dataset_name = 'Scenenet'
#dataset_name = 'TOY'
# dataset_name = 'YCB'
if dataset_name=='VKITTI':
    # trainset = 'trainval'
    # trainset = 'val'

    trainset = 'val'
    valset = 'train'

    # ADD
    ganset = 'val'
    ganset_g = 'train'
    h = 128
    w = 512
    # have_depth_gt = True #???
elif dataset_name=='KITTI':
    trainset = 'train'
    valset = 'val'
    h = 128
    w = 512
elif dataset_name=='KITTI2':
    trainset = 'train'
    valset = 'val'
    h = 128
    w = 512
    have_depth_gt = False
elif dataset_name=='TOY':
    # trainset = 'all'
    # valset = 'all'
    trainset = 'all_hard'
    valset = 'all_hard'
    have_depth_gt = True
    h = 160
    w = 272
elif dataset_name=='YCB':
    trainset = 'train'
    valset = 'val'
    have_depth_gt = True
else:
    trainset = ''

#----------- net design -----------#
do_resnet = False
do_vis = True
do_train = True
do_ft = False
do_crop_aug = False

## odo setup
# CHA
do_depth = True
do_odo = True
do_seg = False

do_flow = False
do_mask = False
do_pose = False
do_com = False

do_train_depth = True # don't do unsup finetuning, unless you handle the scale ambiguity
do_train_odo = True
do_train_flow = False
do_train_mask = False
do_train_pose = False
do_train_com = False
do_train_seg = False

if do_pose or do_com:
    # you need these
    do_mask = True
    do_depth = True
    # they should be pretrained
    do_train_depth = False
    do_train_mask = False
# if do_odo:
#     do_depth=True
#     # do_flow=True

#----------- layers -----------#
nLayers_depth = 4
nLayers_flow = 4
nLayers_odo = 5
nLayers_mask = 1
nLayers_pose = 3
nLayers_com = 2

#----------- general hypers -----------#
bs = 4 # this is overwritten if do_train = False
lr = 1e-6
queue_capacity = 100+3*bs
do_debug = False
do_backward = False
do_unsup = True
use_wrong_f = False
do_fast_debug = False

if do_fast_debug: 
    nLayers_depth = 1
    nLayers_flow = 1
    nLayers_odo = 1
    nLayers_mask = 1
    nLayers_pose = 1
    nLayers_com = 1
    bs = 1
    # lr = 1E-9 #numerical problems happen otherwise ##### why?
    lr = 1e-5
    # h = 128
    # w = 512
    h = 128
    w = 512
    depth_init = ""
    flow_init = ""
    odo_init = ""
    seg_init = ""
    gan_init = ""
    queue_capacity = 8 #so we don't have to wait for queue to fill

# "classic" does true deconv; with this false, we do NN then a conv 
do_classic_deconv = True
# pad = "REFLECT" # seems a bit better. it's also what i would choose in matlab...
pad = "SYMMETRIC" # but there's a ref for this!

#----------- odo hypers -----------#
do_fc = False
# when fully-connected=True, we don't need to mask.
# if false, please set these three...
do_mask_moving_w_gt = False
do_mask_rt = False
do_mask_photo = False

## odo inputs
cat_rgb = True
cat_angles = True
cat_flow = True
cat_ang_flow = True
cat_ang_diff = False
cat_depth = False # please keep this False

rtd_coeff = 0.0
rta_coeff = 0.0 # should be 100x the others, seems like
td_coeff = 0.0
ta_coeff = 0.0
# rtd_coeff = 1.0
# rta_coeff = 100.0 # should be 100x the others, seems like
# td_coeff = 1.0
# ta_coeff = 1.0

odo_photo_coeff = 10.0

#----------- depth hypers -----------#
## back to just L1 now
# # choose one of these
# depth_do_l1 = True
# depth_do_l2 = False
# depth_do_si = False
# depth_do_hu = False

# depth_main_coeff = 0.0
# depth_inval_coeff = 0.0
# ADD

valid = 1
df_dim = 64

depth_coeff = 0.0
depth_main_coeff = 0.0
depth_inval_coeff = 0.0
depth_smooth_coeff = 0.0

# depth_prior = 3.40 # this is the mean GT depth on the val set; roughly 30m
depth_prior = 3.6 # this is the mean GT depth on the train set
depth_prior_coeff = 0.1 # set this to 0 if we're doing sup

#----------- flow hypers -----------#
flow_l2_coeff = 0.0
flow_photo_coeff = 10.0
flow_smooth_coeff = 1.0

#----------- mask hypers -----------#
mask_coeff = 1.0
mask_smooth_coeff = 0.5

#----------- pose hypers -----------#
pose_photo_coeff = 1.0
pose_photo2_coeff = 10.0
pose_smooth_dp_coeff = 0.0 # 1.0
pose_smooth_sine_coeff = 0.0 # 100.0
# masked losses
pose_mphoto_coeff = 1.0
pose_mphoto2_coeff = 10.0
pose_msmooth_dp_coeff = 0.0
pose_msmooth_sine_coeff = 0.0
# extras
pose_depthwarp_coeff = 1000.0
pose_mdepthwarp_coeff = 0.0

#----------- com hypers -----------#
com_photo_coeff = 1
com_photo2_coeff = 10
com_smooth_coeff = 1.0
# masked losses
com_mphoto_coeff = 0
com_mphoto2_coeff = 0
com_msmooth_coeff = 10.0
# extras
com_depthwarp_coeff = 1.0
com_mdepthwarp_coeff = 0.0

com_centroid_coeff = 0.0
com_spread_coeff = 0.0
com_zerosum_coeff = 10.0

#----------- seg hypers -----------#
seg_coeff = 1.0
# Setting 14 for resnet but for fcn it has to be 13
num_seg_classes = 14
#num_seg_classes = 20
IGNORE_LABEL = 255
not_restore_last = False

#----------------- gan hypers-------------#
do_seg_gan = False
do_seg_gan_original = False
do_depth_gan = False
do_depth_gan_original = False
do_gan = do_seg_gan or do_depth_gan
gan_lr = 1e-6

#----------------------+

# INPUT_SIZE = '%d,%d' % (h,w)
# LEARNING_RATE = 2.5e-3
# MOMENTUM = 0.9
# NUM_CLASSES = 14
# NUM_STEPS = 20001
# POWER = 0.9
# RANDOM_SEED = 1234

# RESTORE_FROM = path + 'deeplab_resnet.ckpt'

# SAVE_NUM_IMAGES = BATCH_SIZE
# SAVE_PRED_EVERY = 100
# SAVE_MODEL_EVERY = 1000

# # SNAPSHOT_DIR = path + 'snapshots/'
# SNAPSHOT_DIR = "%s/snapshots/%02dx%dx%d_%.1e" % (path,BATCH_SIZE,h,w,LEARNING_RATE)
seg_weight_decay = 0.0005
save_num_images = 1



#----------- rules ----------#
do_object_depth_rule = False #requires VKITTI!
do_bbox2cuboid = True
object_depth_rule_coeff = 1.0

do_normals_rule = False
normals_smooth_coeff = 0.1
normals_roadup_coeff = 1.0

do_mask_rcnn = False
do_mask_rcnn2 = False

gt_box_source = 'objs' #'segs'

#----------- mod -----------#
mod = "bench_debug"

############ slower-to-change hyperparams below here ############

eps = 1e-6

## logging
log_freq_t = 1
log_freq_v = 1
dump_freq = 1
snap_freq = 1
print_loss = True

if do_fast_debug:
    log_freq_t = 1
    log_freq_v = 1
    dump_freq = 10
    snap_freq = 10
    
if do_train:
    # watch out, we may override these based on the dataset
    max_iters = 200000
    shuffle_val = True
else:
    bs = 1
    do_backward = False
    # max_iters = nVal/bs
    # max_iters = 3000
    max_iters = 2670
    shuffle_val = False
    log_freq_v = 1
    dump_freq = 100

if dataset_name == 'VKITTI':
    fy = 725.0
    fx = 725.0
    y0 = 187.0
    x0 = 620.5
    root_dir = '/home/ishu/'
    dataset_location = root_dir + "datasets/vkitti/singles_w_everything6/"
    dataset_t = root_dir + "datasets/vkitti/lists/%s.txt" % trainset
    dataset_v = root_dir + "datasets/vkitti/lists/%s.txt" % valset
    # ADD
    dataset_gan = root_dir + "datasets/vkitti/lists/%s.txt" % ganset
    dataset_gan_g = root_dir + "datasets/vkitti/lists/%s.txt" % ganset_g
    scale = 0.5
    if do_train:
        max_iters = 200000
        if do_ft:
            max_iters = 400000
elif dataset_name == 'Scenenet':
    fy = 289.71
    fx = 277.128
    y0 = 119.5
    x0 = 159.5
    dataset_t = "datasets/scenenet/train.txt"
    dataset_v = dataset_t
    dataset_location = "datasets/scenenet/tfrs/"
    scale = 1.0
    seg_coeff = 0.0
elif dataset_name == 'KITTI':
    dataset_location = "datasets/kitti_odometry/records/"
    dataset_t = "datasets/kitti_odometry/%s.txt" % trainset
    dataset_v = "datasets/kitti_odometry/%s.txt" % valset
    scale = 0.5

    # these are actually for vkitti. only use these if "use_wrong_f=True"
    fy = 725.0
    fx = 725.0
    y0 = 187.0
    x0 = 620.5
    
elif dataset_name == 'KITTI2':
    dataset_location = "datasets/kitti_odometry/records3/"
    dataset_t = "datasets/kitti_odometry/%s.txt" % trainset
    dataset_v = "datasets/kitti_odometry/%s.txt" % valset
    scale = 0.5
    # these are actually for vkitti. only use these if "use_wrong_f=True"
    fy = 725.0
    fx = 725.0
    y0 = 187.0
    x0 = 620.5
    # these are true in vkitti for at least one seq
    # fy = 718.0
    # fx = 718.0
    # y0 = 185.0
    # x0 = 607.0
    if do_train:
        max_iters = 200000
        if do_ft:
            max_iters = 400000
elif dataset_name == 'TOY':
    fy = 525.0
    fx = 525.0
    scale = 0.5
    dataset_t = "datasets/toy/%s.txt" % trainset
    dataset_v = "datasets/toy/%s.txt" % valset
    dataset_location = "datasets/toy/records_hard/"
    if do_train:
        max_iters = 100000
elif dataset_name == 'YCB':
    scale = 0.5
    dataset_t = "datasets/ycb/%s.txt" % trainset
    dataset_v = "datasets/ycb/%s.txt" % valset
    dataset_location = "datasets/ycb/"

    
############ autogen a name ############

name = "%02dx%dx%d_%.1e" % (bs,h,w,lr)

if do_unsup: 
    # for safety: set the supervised losses to 0
    flow_l2_coeff = 0.0
    depth_main_coeff = 0.0
    depth_inval_coeff = 0.0
    seg_coeff = 0.0
    rtd_coeff = 0.0
    rta_coeff = 0.0
    td_coeff = 0.0
    ta_coeff = 0.0
    mask_coeff = 0.0
    
if do_depth:
    if do_train_depth and do_unsup:
        name = "%s_D%d" % (name,
                           nLayers_depth)
        name = "%s_s%.1f_p%.1f" % (name, depth_smooth_coeff, depth_prior_coeff)
    elif do_train_depth:
        name = "%s_D%d_%.1f" % (name,
                                nLayers_depth,
                                depth_main_coeff)
        name = "%s_inval%.2f_s%.1f_p%.1f" % (name, depth_inval_coeff, depth_smooth_coeff, depth_prior_coeff)
    else:
        name = "%s_D%d" % (name,
                           nLayers_depth)
    if do_object_depth_rule:
        name = "%s_obj%.1f" % (name,
                               object_depth_rule_coeff)
    if do_normals_rule:
        name = "%s_surf%.1f_up%.1f" % (name,
                                       normals_smooth_coeff,
                                       normals_roadup_coeff)
if do_mask:
    # this net always has some kind of supervision. total unsup doesn't seem to work.
    if do_train_mask:
        name = "%s_M%d_%.1f_s%.1f" % (name,
                                      nLayers_mask,
                                      mask_coeff,
                                      mask_smooth_coeff)
    else:
        name = "%s_M%d" % (name,
                           nLayers_mask)
if do_flow:
    if do_train_flow and do_unsup:
        name = "%s_F%d_p%.1f_s%.1f" % (name,
                                       nLayers_flow,
                                       flow_photo_coeff,
                                       flow_smooth_coeff)
    elif do_train_flow:
        name = "%s_F%d_l%.2f_p%.1f_s%.1f" % (name,
                                             nLayers_flow,
                                             flow_l2_coeff,
                                             flow_photo_coeff,
                                             flow_smooth_coeff)
    else:
        name = "%s_F%d" % (name,
                           nLayers_flow)
if do_odo:
    if do_train_odo and do_unsup:
        name = "%s_O%d_p%.2f" % (name,
                                 nLayers_odo,
                                 odo_photo_coeff)
    elif do_train_odo:
        name = "%s_O%d_p%.2f_rtd%.1f_rta%.1f_td%.1f_ta%.1f" % (name,
                                                               nLayers_odo,
                                                               odo_photo_coeff,
                                                               rtd_coeff,
                                                               rta_coeff,
                                                               td_coeff,
                                                               ta_coeff)
    else:
        name = "%s_O%d" % (name,
                           nLayers_odo)
    # for odo, i'm still figuring out the best set of inputs
    if cat_rgb:
        name = "%s+r" % name
    if cat_angles:
        name = "%s+a" % name
    if do_flow:
        if cat_flow:
            name = "%s+f" % name
        if cat_ang_flow:
            name = "%s+af" % name
        if cat_ang_diff:
            name = "%s+ad" % name
    if cat_depth:
        name = "%s+d" % name
    if do_fc:
        name = "%s_fc" % name
    else:
        if do_mask_rt or do_mask_photo:
            if do_mask_rt:
                name = "%s_mrt" % name
            if do_mask_photo:
                name = "%s_mp" % name
            if do_mask_moving_w_gt:
                name = "%s_g" % name
            else:
                name = "%s_e" % name
if do_pose:
    if do_train_pose:
        name = "%s_P%d_p%.1f_p%.1f_m%.1f_m%.1f_dp%.1f_sine%.1f_mdp%.1f_msine%.1f_z%.1f_mz%.1f" % (name,
                                                                                                  nLayers_pose,
                                                                                                  pose_photo_coeff,
                                                                                                  pose_photo2_coeff,
                                                                                                  pose_mphoto_coeff,
                                                                                                  pose_mphoto2_coeff,
                                                                                                  pose_smooth_dp_coeff,
                                                                                                  pose_smooth_sine_coeff,
                                                                                                  pose_msmooth_dp_coeff,
                                                                                                  pose_msmooth_sine_coeff,
                                                                                                  pose_depthwarp_coeff,
                                                                                                  pose_mdepthwarp_coeff)
    else:
        name = "%s_P%d" % (name,
                           nLayers_pose)
if do_com:
    # dp for delta com
    if do_train:
        name = "%s_C%d_p%.1f_p%.1f_m%.1f_m%.1f_c%.1f_mc%.1f_z%.1f_mz%.1f_c%.1f_s%.1f_z%.1f" % (name,
                                                                                               nLayers_com,
                                                                                               com_photo_coeff,
                                                                                               com_photo2_coeff,
                                                                                               com_mphoto_coeff,
                                                                                               com_mphoto2_coeff,
                                                                                               com_smooth_coeff,
                                                                                               com_msmooth_coeff,
                                                                                               com_depthwarp_coeff,
                                                                                               com_mdepthwarp_coeff,
                                                                                               com_centroid_coeff,
                                                                                               com_spread_coeff,
                                                                                               com_zerosum_coeff)
    else:
        name = "%s_C%d" % (name,
                           nLayers_com)
if do_seg:
    name = "%s_S" % (name)

if do_gan:
    name = "%s_G%f_%s_" % (name, gan_lr, ganset)


if do_seg_gan:
    name = "%s_SG_%d_" % (name, do_seg_gan_original)

if do_depth_gan:
    name = "%s_DG_%d_" % (name, do_depth_gan_original)
##### end model description

if do_backward:
    name = "%s_backward" % name
if use_wrong_f:
    name = "%s_wf" % name

# if not do_train_depth:
#     name = "%s_ntd" % name
# if not do_train_flow:
#     name = "%s_ntf" % name
# if not do_train_mask:
#     name = "%s_ntm" % name
# if not do_train_odo:
#     name = "%s_nto" % name

# if do_classic_deconv:
#     name = "%s_classic" % name

# add some training data info
name = "%s_%s" % (name, dataset_name)
if trainset:
    name = "%s_%s" % (name, trainset)

if mod:
    name = "%s_%s" % (name, mod)

# if we're testing, forget the crazy name, just call it init_mod
if not do_train:
    name = "TEST_%s_%dx%d_%s_%s_%s_TEST" % (total_init, h, w, dataset_name, valset, mod)

# if we're testing, turn on the modules asked for in the model name
if not do_train:
    if total_init.find("D") == -1:
        do_depth = False
    else:
        do_depth = True
    if total_init.find("M") == -1:
        do_mask = False
    else:
        do_mask = True
    if total_init.find("F") == -1:
        do_flow = False
    else:
        do_flow = True
    if total_init.find("O") == -1:
        do_odo = False
    else:
        do_odo = True
    if total_init.find("P") == -1:
        do_pose = False
    else:
        do_pose = True
    if total_init.find("C") == -1:
        do_com = False
    else:
        do_com = True
    
# helpers
if not do_depth:
    depth_init = ""
if not do_flow:
    flow_init = ""
if not do_odo:
    odo_init = ""
if not do_seg:
    seg_init = ""
if not do_gan:
    gan_init = ""

name = "_%s_TRAIN_%s" % (name, trainset)

if do_gan:
    name = "_%s_GTRAIN_%s" % (name, ganset)

print name