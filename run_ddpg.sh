#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "DDPG_VSS5V5_0$i" -e VSS5V5-v0
done
