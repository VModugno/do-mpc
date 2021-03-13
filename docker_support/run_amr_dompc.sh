#!/bin/bash
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DO_MPC_REPO=amr_dompc_repo
PROJECT_DIR=project_source
if [ "$1" == "hostrepo" ]; then
    ABS_PATH_DOMPCREPO=$(readlink -f $THIS_SCRIPT_DIR/..)
    MOUNT_HOSTREPO="-v $ABS_PATH_DOMPCREPO:/workdir/amr_dompc/do-mpc:rw"
else
    MOUNT_HOSTREPO=  
fi
CONDA_ENV="dompc_dev" 
xhost local:root &>/dev/null
XAUTH_FILE=$(mktemp)
docker run -ti --rm --network=host \
        --cap-add=IPC_LOCK \
        $MOUNT_HOSTREPO \
        --ipc=host \
        -v $XAUTH_FILE:$XAUTH_FILE:rw \
        -e XAUTHORITY=$XAUTH_FILE \
        --device /dev/dri \
        -e DISPLAY=unix$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DEFAULT_CONDA_ENV=$CONDA_ENV \
        amr_dompc
