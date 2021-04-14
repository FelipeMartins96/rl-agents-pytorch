#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg_ma.py --cuda -n "DDPG_VSSMA-v0_0$i" -e VSSMA-v0
done
