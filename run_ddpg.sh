#!/bin/bash
#for i in 0 1 2 3 4; do
python train_ddpg.py --cuda -n "kick_dist_ta_without_energy" -e SSLGoToBallShoot-v0
#done
