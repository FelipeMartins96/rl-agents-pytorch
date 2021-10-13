#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "VSS_A_rsimTuneTorque" -e VSS_A_rsim-v0
done
