#!/bin/bash
for i in 0 1 2 3 4; do
  python train_sac.py --cuda -n "SAC_SSLStaticDefenders-v0_0$i" -e SSLStaticDefenders-v0
  python train_ddpg.py --cuda -n "DDPG_SSLStaticDefenders-v0_0$i" -e SSLStaticDefenders-v0
done
