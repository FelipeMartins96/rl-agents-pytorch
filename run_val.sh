EXP_NAME="rSoccer-paper-val"
EXP_ARGS=""
case $1 in
0) python validate.py -n sa;;
1) python validate.py -n rsa;;
2) python validate.py -n il;;
3) python validate.py -n jal;;
*) echo "Opcao Invalida!" ;;
esac