#!/bin/bash
set -ex
echo 'Welcome: now the amr version of dompc will be installed' 
conda run -n dompc_dev /bin/bash -c "pip3 install -e /workdir/amr_repo/do-mpc"

#This section is to use xvfb by default in case of need of an xserver (see baseline-zoo entrypoint.sh)
# Set up display; otherwise rendering will fail
Xvfb :1 -screen 0 1024x768x24 &

if [ "x$DISPLAY" = "x" ]; then
  export DISPLAY=:1
fi
# Wait for the file to come up
display=1
file="/tmp/.X11-unix/X$display"

sleep 1

for i in $(seq 1 10); do
    if [ -e "$file" ]; then
	     break
    fi

    echo "Waiting for $file to be created (try $i/10)"
    sleep "$i"
done
if ! [ -e "$file" ]; then
    echo "Timing out: $file was not created"
    exit 1
fi

echo "DISPLAY is $DISPLAY"

exec "$@"
