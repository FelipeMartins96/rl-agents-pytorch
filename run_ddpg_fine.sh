#!/bin/bash
for i in 0; do
  python train_ddpg_fine.py --cuda -n "FIRA_fine_atkr_sim-$i" -e VSS_B-v0 -c ./saves/atk_rsim.pth
done
