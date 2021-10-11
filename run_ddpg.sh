#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "vss_B_rsim" -e VSS_B_rsim-v0
done
