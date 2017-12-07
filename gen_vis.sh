#!/bin/bash

python gen_vis.py


# MODEL="*zero"
# MODEL="*_forreal"
# #MODEL="lr_0.0001_bs_64_hw_128_128_depth_1.0_rdist_1.0_rang_100.0_photo_0.00_nLayersD_4_nLayersE_4_balanced_super_fb"
# # MODEL="*with*"
# MODEL="32x128x128_1.0e-04_fl0.00_fp10.0_fs1.0_d1.0_rtd0.1_rta50.0_ta0.0_tm0.0_op1.00_D3_E3_F3_odo_fb*"
# # MODEL="02*"
# MODEL="*depthOnly"
# MODEL="*load_depth*"


google-chrome-stable vis_$MODEL.html
