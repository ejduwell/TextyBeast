#!/bin/bash

# Written by E.J. Duwell, PhD (8/31/23)
#
# srcBRCfrmBPFL: (source .bashrc from .bash_profile)
# Ensures that a ~/.bashrc file exists, that a ~/.bash_profile
# script exists, and that .bashrc is sourced from .bash_profile.
#
# This script checks if .bashrc and/or .bash_profile exist.
# ------------------------------------------------------------------------
#  - If only .bashrc exists:
#     A ~/.bash_profile file is created
#     A snippet is appended to the end of .bash_profile to source .bashrc
#
#  - If only .bash_profile exists:
#     A ~/.bashrc file is created
#     A snippet is appended to the end of .bash_profile to source .bashrc
#     (if its not already present)
#
#  - If both exist:
#     A snippet is appended to the end of .bash_profile to source .bashrc
#     (if its not already present)
#
#  - If neither exist (unlikely):
#     A ~/.bashrc file is created
#     A ~/.bash_profile file is created
#     A snippet is appended to the end of .bash_profile to source .bashrc
# ------------------------------------------------------------------------
# Purpose: The purpose of this script is to provide a way to ensure that
# users' bash environments behave in a predictable manner regardless of
# whether they are running mac or linux, and regardless of their personal
# manipulations to their bash setup. After running this script, you should
# be able to ensure that the ~/.bashrc file will be sourced on login and
# interactive shell sessions. After running this, everyone will have both
# a ~/.bashrc and a ~/.bash_profile script, and everyone's ~/.bash_profile
# will source the ~/.bashrc. You should, therefore, be able to more safely
# count on variables, path info, etc. set in the .bashrc file being
# available for your package.
#
# In short, the purpose is to make sure that the shit you put in the
# .bashrc get's sourced, and to aleviate complications/confusions
# in this arena.. If the .bash_profile is sourced, but not .bashrc,
# .bashrc will still get sourced... If the the .bashrc gets sourced but
# not the .bash_profile, .bashrc gets sourced... If both get sourced,
# .bashrc gets sourced...

echo ""
echo "Setting up bash environment to ensure ~/.bashrc exists and is sourced on login/interactive bash shells..."
echo ""

# Check if ~/.bashrc exists..
if [ -r ~/.bashrc ]; then
    echo "~/.bashrc exists ..."
    echo ""
    # if ~/.bashrc exists..
    brcExisted=1

else
    echo "~/.bashrc did not exist ..."
    echo "creating it now ..."
    echo ""
    # if ~/.bashrc does not exist..
    brcExisted=0
    # create an empty ~/.bashrc
    touch ~/.bashrc
    # make sure it is readable..
    chmod +r ~/.bashrc
fi

# Check if ~/.bash_profile exists..
if [ -r ~/.bash_profile ]; then
    echo "~/.bash_profile file exists ..."
    echo ""
    
    # If ~/.bash_profile exists:
    #---------------------------------
    bpfExisted=1
    # Check if the code snippet already exists in the .bash_profile
    grep 'source ~/.bashrc' ~/.bash_profile > /dev/null

    # If the snippet is not present, append it
    if [ $? -ne 0 ]; then
	echo "=============================================" >> ~/.bash_profile
	echo "# Added by TextyBeast to" >> ~/.bash_profile
	echo "# ensure .bashrc is sourced" >> ~/.bash_profile
	echo "source ~/.bashrc" >> ~/.bash_profile
	echo "=============================================" >> ~/.bash_profile
	echo "Snippet: 'source ~/.bashrc' added to .bash_profile!"
	echo ""
    else
	echo "Snippet: 'source ~/.bashrc' already existed in .bash_profile!"
	echo ""
    fi

else
    echo "~/.bash_profile file does not exist ..."
    echo "creating it now ..."
    echo ""
    # If ~/.bash_profile does not exist:
    #---------------------------------
    bpfExisted=0
    # create an empty ~/.bash_profile
    touch ~/.bash_profile
    # make sure it is readable..
    chmod +r ~/.bash_profile
    # Add the commands
    echo "=============================================" >> ~/.bash_profile
    echo "# Added by TextyBeast to" >> ~/.bash_profile
    echo "# ensure .bashrc is sourced" >> ~/.bash_profile
    echo "source ~/.bashrc" >> ~/.bash_profile
    echo "=============================================" >> ~/.bash_profile   
    echo "Snippet: 'source ~/.bashrc' added to .bash_profile!"
    echo ""
fi
