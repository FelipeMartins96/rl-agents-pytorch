EXP_NAME="rsoccer-w4"
EXP_ARGS=""
case $1 in
0) xvfb-run -a python train_ddpg.py    --cuda -n W4-SA-DDPG   -e VSS-v0;;
1) xvfb-run -a python train_ddpg.py    --cuda -n W4-JAL-DDPG  -e VSSJAL-v0;;
2) xvfb-run -a python train_ddpg_ma2.py --cuda -n W4-IL-MADDPG   -e VSSMA-v0;;
*) echo "Opcao Invalida!" ;;
esac