#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "vss_BRW_rsim_v1" -e VSS_BRW_rsim-v1
done
