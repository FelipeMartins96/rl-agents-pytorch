#!/bin/bash
for i in 0; do
  python train_sac_ma.py -n "SAC_VSSMA-v0_0$i" -e VSSMA-v0
done
