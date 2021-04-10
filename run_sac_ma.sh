#!/bin/bash
for i in 0; do
  python train_sac_ma.py --cuda -n "SAC_SSLPassEndurance-v0_0$i" -e SSLPassEndurance-v0
done
