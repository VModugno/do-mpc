#!/bin/bash
HOST_REPO=0                                  # Mount the amr_dompc local repo
NVIDIA_CONTAINER=0                            # Run the docker machine with nvidiadocker to use host xserver
usage() {                                    # Function: Print a help message.
  echo "Usage: $0 [ -r ] [ -n ]"
  echo "-r if the local amr repo has to be mounted"
  echo "-n if in graphical mode on a nvidia host"
}
exit_abnormal() {                         # Function: Exit with error.
  usage
  exit 1
}
while getopts "rnh" options; do            
                        
  case "${options}" in                    
    r)                                    
      HOST_REPO=1                      
      ;;
    n)                                   
      NVIDIA_CONTAINER=1
      ;;
    h)
      usage
      exit 0
      ;;
    *)                                    # If unknown (any other) option:
      exit_abnormal                       # Exit abnormally.
      ;;
  esac
done

if [ $HOST_REPO -eq 1 ]; then                 
     echo "We want to mount local repo" 
     THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
     ABS_PATH_DOMPCREPO=$(readlink -f $THIS_SCRIPT_DIR/../..)
     MOUNT_HOSTREPO="-v $ABS_PATH_DOMPCREPO:/workdir/amr_repo:rw"
else                                      
  echo "We will use the repo in the docker image"  
  MOUNT_HOSTREPO= 
fi

DOCKER_COMMAND=docker

if [ $NVIDIA_CONTAINER -eq 1 ]; then
    echo "We want to launch an nvidia "
    DOCKER_COMMAND=nvidia-docker
    DISPLAY_VALUE=$DISPLAY
else
    echo "We want to use a headless container"
    DISPLAY_VALUE=:1
fi

CONDA_ENV="dompc_dev" 
xhost +local:root &>/dev/null
XAUTH_FILE=$(mktemp)
$DOCKER_COMMAND run -ti --rm --network=host \
        --cap-add=IPC_LOCK \
        $MOUNT_HOSTREPO \
        --ipc=host \
        -v $XAUTH_FILE:$XAUTH_FILE:rw \
        -e XAUTHORITY=$XAUTH_FILE \
        -e DISPLAY=$DISPLAY_VALUE \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DEFAULT_CONDA_ENV=$CONDA_ENV \
        amr_dompc
        
#--device /dev/dri 


#nvidia-docker run -it \
#    --env="DISPLAY" \
#    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#    glxgears_test glxgears
