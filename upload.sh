#!/bin/bash


rsync -vaP -e "ssh" *.py matrix.ml.cmu.edu:~/dlslam/
rsync -vaP -e "ssh" ishu_resnet matrix.ml.cmu.edu:~/dlslam/
rsync -vaP -e "ssh" rcnn matrix.ml.cmu.edu:~/dlslam/

# rsync -vaP -e "ssh" tf_faster_rcnn matrix.ml.cmu.edu:~/dlslam/
# rsync -vaP -e "ssh" FastMaskRCNN matrix.ml.cmu.edu:~/dlslam/
# rsync -vaP -e "ssh" *.py bash.autonlab.org:~/dlslam/
# rsync -vaP -e "ssh" write_single_files.py bash.autonlab.org:~/dlslam/
# rsync -vaP -e "ssh" log matrix.ml.cmu.edu:~/dlslam/
# rsync -vaP -e "ssh" resnet* matrix.ml.cmu.edu:~/dlslam/
# rsync -vaP -e "ssh" vgg16.npy matrix.ml.cmu.edu:~/dlslam/
# rsync -vaP -e "ssh" deeplab_resnet matrix.ml.cmu.edu:~/dlslam/

# rsync -vaP -e "ssh" poses_m_lr_0.00* bash.autonlab.org:~/dlslam_d/

# rsync -vaP -e "ssh" resnet_* matrix.ml.cmu.edu:~/dlslam/
