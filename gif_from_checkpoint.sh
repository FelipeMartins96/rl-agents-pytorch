if [ -z $1 ]
then
    echo "sh train_script.sh [dir]"
else
    for instance in `/bin/ls -d $1*`; do 
        echo $instance
        xvfb-run -s "-screen 0 1400x900x24" python generate_gifs.py -c ${instance} --cuda
    done
fi