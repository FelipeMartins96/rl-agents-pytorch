EXP_NAME="rSoccer-paper"
EXP_ARGS=""
case $1 in
0) xvfb-run -a python train_ddpg.py    --cuda -n $EXP_NAME-SA   -e VSS-v0;;
1) xvfb-run -a python train_ddpg.py    --cuda -n $EXP_NAME-JAL  -e VSSJAL-v0;;
2) xvfb-run -a python train_ddpg_ma.py --cuda -n $EXP_NAME-IL   -e VSSMA-v0;;
3) xvfb-run -a python train_ddpg_ma2.py --cuda -n $EXP_NAME-IL-MADDPG   -e VSSMA-v0;;
*) echo "Opcao Invalida!" ;;
esac