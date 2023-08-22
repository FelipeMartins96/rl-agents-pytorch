EXP_NAME="rsoccer-w3"
EXP_ARGS=""
case $1 in
0) xvfb-run -a python train_ddpg.py    --cuda -n W3-SA-DDPG   -e VSS-v0;;
1) xvfb-run -a python train_ddpg.py    --cuda -n W3-JAL-DDPG  -e VSSJAL-v0;;
2) xvfb-run -a python train_ddpg_ma2.py --cuda -n W3-IL-MADDPG   -e VSSMA-v0;;
*) echo "Opcao Invalida!" ;;
esac