SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ "$3" = "" ]
then
    echo "Arguments insufficient"
    exit 1
fi

GPU_ID=$1
CHECKPOINT=$2

docker run --rm --gpus "device=${GPU_ID}" \
        -v /etc/localtime:/etc/localtime:ro \
        -v $SCRIPT_DIR:/workspace \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -u $(id -u $USER):$(id -g $USER) \
        cgdcase:0.2 python3 adversarial_attack.py --resume $2 --eid $3 >> err.log &
