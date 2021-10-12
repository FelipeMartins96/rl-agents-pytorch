#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "vss_BRW_rsim_v1" -e VSS_BRW_rsim-v1
  python train_ddpg.py --cuda -n "vss_BRW_rsim_v2" -e VSS_BRW_rsim-v2
done
