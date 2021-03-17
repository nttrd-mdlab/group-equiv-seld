SCRIPT_DIR=$(cd $(dirname $0); pwd)
GPU_ID=$1
if [ "$1" = "" ]
then
    echo "Error: no GPU assigned."
    exit 1
fi
docker run --rm --gpus "device=${GPU_ID}" \
        -v /etc/localtime:/etc/localtime:ro \
        -v $SCRIPT_DIR:/workspace \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -u $(id -u $USER):$(id -g $USER) \
        cgdcase:0.2 python3 main.py >> err.log &
