#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg.py --cuda -n "DDPG_SSLGoToBallIR-v1_0$i" -e SSLGoToBallIR-v1
done
