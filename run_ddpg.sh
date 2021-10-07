#!/bin/bash
#for i in 0 1 2 3 4; do
python train_ddpg.py --cuda -n "all_std_0.99_0.999_12_5_3_rw_bg_bd_200_v0" -e SSLGoToBallShoot-v0 
#done
