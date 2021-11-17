#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "pytorch-baseline-n1-p1-5-s.4-t.15" -e VSS-v0
done
