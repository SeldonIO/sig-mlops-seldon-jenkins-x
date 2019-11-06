#!/bin/bash

set -o errexit
set -o pipefail

# ENSURE WE ARE IN THE DIR OF SCRIPT
cd -P -- "$(dirname -- "$0")" 
# SO WE CAN MOVE RELATIVE TO THE ACTUAL BASE DIR
cd ../../

# FIRST WE START THE DOCKER DAEMON
service docker start
# the service can be started but the docker socket not ready, wait for ready
WAIT_N=0
while true; do
    # docker ps -q should only work if the daemon is ready
    docker ps -q > /dev/null 2>&1 && break
    if [[ ${WAIT_N} -lt 5 ]]; then
        WAIT_N=$((WAIT_N+1))
        echo "[SETUP] Waiting for Docker to be ready, sleeping for ${WAIT_N} seconds ..."
        sleep ${WAIT_N}
    else
        echo "[SETUP] Reached maximum attempts, not waiting any longer ..."
        break
    fi
done

#######################################
# AVOID EXIT ON ERROR FOR FOLLOWING CMDS
set +o errexit

if [ -z $DOCKER_AUTH_CONFIG ]; then
    echo "DOCKER AUTH NOT SET"
    DOCKER_CMD_VALUE=1
else
    make \
        build \
        push_to_dockerhub
    DOCKER_CMD_VALUE=$?
fi


#######################################
# EXIT STOPS COMMANDS FROM HERE ONWARDS
set -o errexit

# CLEANING DOCKER
docker ps -aq | xargs -r docker rm -f || true
service docker stop || true

# NOW THAT WE'VE CLEANED WE CAN EXIT ON TEST EXIT VALUE
exit ${DOCKER_CMD_VALUE}

