#!/bin/bash

# READ IN INPUTS FROM USER
echo "Is your local machine running MacOSX or a Linux distro?"
echo "Press 'm' for Mac or 'l' for Linux. Then press enter/return:"
read osType
echo ""

if [[ $osType == "l" ]]; then
rcFile=~/.bashrc
fi

if [[ $osType == "m" ]]; then
rcFile=~/.bash_profile
fi

echo "Adding TextyBeastLocal folder path to permanent path"
echo "by adding the following commands to the end of $rcFile:"
echo "------------------------------------------------------------"
# First Add Comment In .bashrc/.bash_profile..
echo "">> $rcFile
echo "# Add TextyBeastLocal to Path to Make Functions Callable Everywhere:">> $rcFile
echo "# ---------------------------------------------------------">> $rcFile

#Save current path location (this script should be in the TextyBeastLocal folder...)
txtyBstLocalPath=$(pwd)

# build and add export command to export txtyBstLocalPath from bashrc file:
Cmd1_1='export PATH='
Cmd1=$Cmd1_1'"'$txtyBstLocalPath":$""PATH"'"'
echo $Cmd1 >> $rcFile
echo $Cmd1

echo ""
echo "Run: source $rcFile to update path in this terminal session..."
echo ""
