screen -S robocinrl python train_sac.py --cuda -n SAC_VSS-v0_00 &&
screen -S robocinrl python train_sac.py --cuda -n SAC_VSS-v0_01 &&
screen -S robocinrl python train_sac.py --cuda -n SAC_VSS-v0_02 &&
screen -S robocinrl python train_sac.py --cuda -n SAC_VSS-v0_03 &&
screen -S robocinrl python train_sac.py --cuda -n SAC_VSS-v0_04
