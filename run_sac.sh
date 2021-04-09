#!/bin/bash
for i in 0 1 2 3 4; do
  python train_sac.py --cuda -n "SAC_SSLGoToBallIR-v1_0$i" -e SSLGoToBallIR-v1
done
