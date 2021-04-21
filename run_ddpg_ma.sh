#!/bin/bash
for i in 0; do
  python train_ddpg_ma.py --cuda -n "DDPG_SSLPassEnduranceMA-v0_0$i" -e SSLPassEnduranceMA-v0
done
