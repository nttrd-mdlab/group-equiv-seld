docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg HTTP_PROXY=$HTTP_PROXY \
    --build-arg HTTPS_PROXY=$HTTPS_PROXY \
    -t cgdcase:0.2 .
