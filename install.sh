#!/bin/bash

# This script is for installation on the remote machine..

# make scripts executable
chmod +wrx makeExctbl; ./makeExctbl

# Detect OS and default shell
userOs=$(./checkOS.sh)


# If remote machine is running macOS (Darwin), run ./buildEnvs_Mac.sh
if [[ $userOs == "Darwin" ]]; then
    echo ""
    echo "Remote machine is running macOS ..."
    echo "Running the Mac Install Procedure ..."
    echo ""
    ./buildEnvs_Mac.sh
fi


# If remote machine is running Linux, run the ./buildEnvs.sh
if [[ $userOs == "Linux" ]]; then
    echo ""
    echo "Remote Machine is running Linux ..."
    echo "Running the Linux Install Procedure ..."
    echo ""
    ./buildEnvs.sh
fi

# If user is not running linux or mac, they are out of luck..
# notify them of the issue on command line..
if [[ $userOs != "Darwin" ]] && [[ $userOs != "Linux" ]]; then
    echo ""
    echo "Remote machine is not running macOS or Linux ..."
    echo "Sorry, OS not supported ..."
    echo ""
fi
