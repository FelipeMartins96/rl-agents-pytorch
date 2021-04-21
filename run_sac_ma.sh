#!/bin/bash
for i in 0; do
  python train_sac_ma.py --cuda -n "SAC_SSLPassEnduranceMA-v0_0$i" -e SSLPassEnduranceMA-v0
done
