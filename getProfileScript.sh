#!/bin/bash

# This Script Detects the User's OS and Default Shell and Returns the Corresponding Profile Script..

#Get Path Info:
startDir=$(pwd)

# get the full path to this script's location on this machine
SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPTDIR # make sure we are currently in that directory...

# * NOTE: assuming that other dependent scripts are present in the same directory*
# * Those include:
#    checkDefaultShell.sh
#    checkOS.sh

# Detect the default shell
defShell=$(./checkDefaultShell.sh)

# Detect the OS
usrOS=$(./checkOS.sh)


case "$defShell" in
    "sh")
        # return shell profile script
        PROFILE_SCRIPT="~/.profile"
        echo "$PROFILE_SCRIPT"
        ;;
        
    "bash")
        # return shell profile script
	# NOTE: TextyBeast is using .bashrc regarless of
	# OS or other setup difs..
	# We're also taking measures to ensure
	# .bashrc is present/sourced..
        PROFILE_SCRIPT="~/.bashrc"
        echo "$PROFILE_SCRIPT"
        ;;
    
    "csh")
        # return shell profile script
        PROFILE_SCRIPT="/etc/csh.cshrc"
        echo "$PROFILE_SCRIPT"
        ;;
    
    "tcsh")
        # return shell profile script
        PROFILE_SCRIPT="~/.tcshrc"
        echo "$PROFILE_SCRIPT"
        ;;
    
    "ksh")
        # return shell profile script
        PROFILE_SCRIPT="~/.profile"
        echo "$PROFILE_SCRIPT"
        ;;
    
    "zsh")
        # return shell profile script
        PROFILE_SCRIPT="~/.zshenv"
        echo "$PROFILE_SCRIPT"
        ;;
    
    "fish")
        # return shell profile script
        PROFILE_SCRIPT="~/.config/fish/config.fish"
        echo "$PROFILE_SCRIPT"
        ;;
    
    *)
        echo "Unsupported Shell: $defShell"
        ;;
esac
