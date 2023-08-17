EXP_NAME="rSoccer-paper"
EXP_ARGS=""
case $1 in
0) xvfb-run -a python train_ddpg.py    --cuda -n W-SA-DDPG   -e VSS-v0;;
1) xvfb-run -a python train_ddpg.py    --cuda -n W-JAL-DDPG  -e VSSJAL-v0;;
3) xvfb-run -a python train_ddpg_ma2.py --cuda -n W-IL-MADDPG   -e VSSMA-v0;;
*) echo "Opcao Invalida!" ;;
esac