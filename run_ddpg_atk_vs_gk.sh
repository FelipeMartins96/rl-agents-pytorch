#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg_atk_vs_gk.py --cuda -n "DDPG_ATKVSGK-v0_0$i" -e VSSATKVSGK-v0 --checkpoint models/atk.pth --checkpoint_gk models/gk_ft2.pth
done
