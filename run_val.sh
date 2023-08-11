EXP_NAME="rSoccer-paper-val"
EXP_ARGS=""
case $1 in
0) python validate.py -n sa-0;;
1) python validate.py -n rsa-0;;
2) python validate.py -n il-ddpg-0;;
3) python validate.py -n il-maddpg-0;;
4) python validate.py -n jal-0;;
5) python validate.py -n sa-1;;
6) python validate.py -n rsa-1;;
7) python validate.py -n il-ddpg-1;;
8) python validate.py -n il-maddpg-1;;
9) python validate.py -n jal-1;;
10) python validate.py -n sa-2;;
11) python validate.py -n rsa-2;;
12) python validate.py -n il-ddpg-2;;
13) python validate.py -n il-maddpg-2;;
14) python validate.py -n jal-2;;
*) echo "Opcao Invalida!" ;;
esac