#!/bin/bash
set -e
echo 'Welcome: now the amr version of dompc will be installed' 
conda run -n dompc_dev /bin/bash -c "pip3 install -e /workdir/amr_dompc/do-mpc"
exec "$@"
