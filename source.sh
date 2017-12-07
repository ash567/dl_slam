#!/bin/bash

# MODEL="sample"
# MODEL="02x128x512_1.0e-04_D4_F4_O6_p10.00+r+a+f+af+ad_backward_VKITTI_train_fixed2"
# MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_F4_O6_p10.00+r+f_backward_VKITTI_train_fixed2"
# MODEL="02x160x272_1.0e-04_D*_1.0_inval0.00_s0.0_p0.0_TOY_all"
# MODEL="*OY_all_hard_aug_regear*"
# MODEL="02x160x272_1.0e-05_D4_M1_P4_p1.0_p10.0_m1.0_m10.0_dp0.0_sine0.0_mdp0.0_msine0.0_z1000.0_mz0.0_backward_TOY_all_hard_regear"
# MODEL="02x160x272_1.0e-04_D4_M1_C*_p1.0_p10.0_m0.0_m0.0_c0.0_mc10.0_z1.0_mz0.0_c0.0_s0.0_z10.0_TOY_all_hard_aug_regear2"
# # MODEL="02x128x512_1.0e-05_D4_s1.0_p1.0_F4_O5_p10.00+r+a+f+af_backward_wf_KITTI2_train_totalunsup_happy"
# MODEL="02x128x512_1.0e-05_D4_s0.1_p1.0_F4_O5_p10.00+r+a+f+af_backward_wf_KITTI2_train_totalunsup_happy"
# # MODEL="*happy"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_O6_p10.00+r_fc_VKITTI_val_trainprior"
# # MODEL="TEST_02x128x512_1.0e-05_D4_0.0_sky0.00_s0.0_p1.0_O6_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r_fc_VKITTI_val_totalunsup_128x512__TEST"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_F4_O6_p10.00+r+f_backward_VKITTI_train_fixed2"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_F4_O6_p10.00+r+f_backward_VKITTI_train_fixed2"
# MODEL="02x160x272_1.0e-04_D4_M1_C*_p1.0_p10.0_m0.0_m0.0_c0.0_mc10.0_z1.0_mz0.0_c0.0_s0.0_z10.0_TOY_all_hard_aug_regear2"
# # MODEL="02x128x512_1.0e-*_D4_s0.0_p1.0_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_bench"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p0.0_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_segtest"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_obj5.0_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_bench"
# # MODEL="02x128x512_1.0e-04_D4_s0.0_p1.0_obj5.0_F4_O5_p10.00+r+a+f+af_backward_VKITTI_train_bench_safer"
# MODEL="02x128x512_1.0e-09_D1_s0.0_p1.0_obj5.0_F1_O1_p10.00+r+a+f+af_backward_VKITTI_train_bench_safer"
# MODEL="02x64x64_1.0e-05_D1_s0.0_p1.0_F1_O1_p10.00+r+a+f+af_backward_VKITTI_train_bench_safer"
# MODEL="01x128x512_1.0e-05_D1_s0.0_p1.0_F1_O1_p10.00+r+a+f+af_backward_VKITTI_train_bench_safer"
# MODEL="01x128x512_1.0e-04_D4_s0.0_p1.0_obj5.0_F4_O5_p10.00+r+a+f+af_VKITTI_train_bench"
# MODEL="01x128x512_1.0e-04_D4_*F4_O5_p10.00+r+a+f+af_VKITTI_train_bench"
# MODEL="01x128x512_1.0e-05_D4_*F4_O5_p10.00+r+a+f+af_VKITTI_train_bench*"
# MODEL="01x128x512_1.0e-05_D4_*F4_O5_p10.00+r+a+f+af_VKITTI_train_bench7"
# MODEL="*bench7"
# MODEL="01x128x512_1.0e-05_D1_s0.1_p0.0_obj1.0_F1_O1_p10.00+r+a+f+af_VKITTI_train_bench7"
# MODEL="01x128x512_1.0e-05_S_VKITTI_train_bench"
# MODEL="16x128x512_1.0e-*_S_VKITTI_train_bench"
# #MODEL="16x128x512_1.0e-*_S_VKITTI_train_bench_debug"
# MODEL="01x128x512_1.0e-05_D4_s0.1_p0.1_obj1.0_VKITTI_train_bench_debug"
# MODEL="16x128x512_1.0e-05_D4_0.0_inval0.00_s0.1_p0.1_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_1.0_inval0.00_s0.1_p0.1_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s0.1_p0.1_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s0.1_p0.1_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s0.0_p0.0_surf0.0_up0.0_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s0.0_p0.0_surf1.0_up0.0_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s1.0_p0.1_surf1.0_up0.0_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_0.0_inval0.00_s1.0_p0.1_surf*_up*_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-04_D4_s1.0_p0.1_surf*_up*_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-0*_D4_s*_p0.1_surf*_up*_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-05_D4_s*_p0.1_surf*_up*_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-05_D4_s*_p0.1_surf*_up*_O5_p10.00+r+a_VKITTI_train_bench_debug"
# MODEL="02x128x512_1.0e-05_D4_s0.0_p0.1_surf*_up*_O5_p10.00+r+a_VKITTI_train_bench_debug"
# MODEL="01x64x64_1.0e-05_D1_0.0_inval0.00_s0.1_p0.1_obj1.0_S_VKITTI_val_bench_debug"
# MODEL="TEST__128x512_VKITTI_val_bench_debug_TEST"
# MODEL="02x128x512_1.0e-05_D4_s0.0_p0.1_surf1.0_up0.0_O5_p10.00+r+a_VKITTI_train_bench_debug"

MODEL="32x128x512_1.0e-04_D4_obj1.0_O5_p10.00_rtd0.0_rta0.0_td0.0_ta0.0+r+a_backward_VKITTI_val_bench_debug"
export MODEL
