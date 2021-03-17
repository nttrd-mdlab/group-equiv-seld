SCRIPT_DIR=$(cd $(dirname $0); pwd)
docker run -d --name tensorboard --rm -p 6006:6006 \
        -v /etc/localtime:/etc/localtime:ro \
        -v $SCRIPT_DIR:/workspace \
        cgdcase:0.2 tensorboard --logdir="tbx" --bind_all >> tensorboard.log &
