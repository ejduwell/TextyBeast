#!/bin/bash

echo ""
echo "Welcome!"
echo ""
echo "This is the primary installation script for TextyBeast."
echo "It should detect the OS and default UNIX shell of the user"
echo "and run the appropriate installation procedure.."

# Get the full path to this script and ensure we start within that directory
SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPTDIR

# Make necessary scripts executable..
chmod +wrx makeExctbl; ./makeExctbl
cd textyBeastLocal
chmod +wrx *
cd $SCRIPTDIR

# Detect OS and default shell
userOs=$(./checkOS.sh)
defShell=$(./checkDefaultShell.sh)

echo ""
echo "Detected that the current default shell is $defShell ..."

# If default shell is bash, ensure ~/.bashrc
# exists and gets sourced on login and interactive shells..
if [[ $defShell == "bash" ]]; then
    ./srcBRCfrmBPFL.sh
fi

# If user is running macOS (Darwin), run the MAC_INSTALL.sh
if [[ $userOs == "Darwin" ]]; then
    echo ""
    echo "Detected the user is running macOS ..."
    echo "Running the Mac Install Procedure ..."
    echo ""
    ./MAC_INSTALL.sh
fi


# If user is running Linux, run the LINUX_INSTALL.sh
if [[ $userOs == "Linux" ]]; then
    echo ""
    echo "Detected the user is running Linux ..."
    echo "Running the Linux Install Procedure ..."
    echo ""
    ./LINUX_INSTALL.sh
fi

# If user is not running linux or mac, they are out of luck..
# notify them of the issue on command line..
if [[ $userOs != "Darwin" ]] && [[ $userOs != "Linux" ]]; then
    echo ""
    echo "User is not running macOS or Linux ..."
    echo "Sorry, OS not supported ..."
    echo ""
fi
