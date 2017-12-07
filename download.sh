#!/bin/bash


# rsync -vaP -e "ssh" matrix.ml.cmu.edu:~/dlslam/vis/$MODEL vis/
rsync -vaP -e "ssh" matrix.ml.cmu.edu:~/dlslam/*jpg .


# rsync -vaP -e "ssh" matrix.ml.cmu.edu:~/dlslam/rcnn .
