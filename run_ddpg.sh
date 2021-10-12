#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "vss_BRW_rsim_v3_halfgrad" -e VSS_BRW_rsim-v3
  python train_ddpg.py --cuda -n "vss_BRW_rsim_v4_halfmove" -e VSS_BRW_rsim-v4
done
