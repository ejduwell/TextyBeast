#!/bin/bash

# data scraping whisper and pyannote for diarized transcription..

#local

# HEADER STUFF
# ====================================================
echo ""
echo "Starting at $(date)"
echo ""
echo "#########################################################################"
echo "#########################################################################"
echo "##                                                                     ##"
echo "##        ::::::::::::::::::::::::    :::::::::::::::::   :::          ##" 
echo "##           :+:    :+:       :+:    :+:    :+:    :+:   :+:           ##"  
echo "##          +:+    +:+        +:+  +:+     +:+     +:+ +:+             ##"    
echo "##         +#+    +#++:++#    +#++:+      +#+      +#++:               ##"      
echo "##        +#+    +#+        +#+  +#+     +#+       +#+                 ##"        
echo "##       #+#    #+#       #+#    #+#    #+#       #+#                  ##"         
echo "##      ###    #############    ###    ###       ###                   ##"          
echo "##            ::::::::: ::::::::::    :::     :::::::::::::::::::      ##" 
echo "##           :+:    :+::+:         :+: :+:  :+:    :+:   :+:           ##"      
echo "##          +:+    +:++:+        +:+   +:+ +:+          +:+            ##"       
echo "##         +#++:++#+ +#++:++#  +#++:++#++:+#++:++#++   +#+             ##"        
echo "##        +#+    +#++#+       +#+     +#+       +#+   +#+              ##"        
echo "##       #+#    #+##+#       #+#     #+##+#    #+#   #+#               ##"         
echo "##      ######### #############     ### ########    ###                ##"    
echo "##                                                                     ##"
echo "##                                                                     ##"
echo "#########################################################################"
echo "#########################################################################"
echo "################ MAKE DIARIZED INTERVIEW TRANSCRIPTS ####################"
echo "################# LOCALLY WITH PYANNOTE AND WHISPER #####################"
echo "#########################################################################"
echo ""


# PROCESS INPUT ARGUMENTS
# ====================================================
# number of expected input arguments (must update if more are added in development):
nxArg=3

# Check that expected number of input args were provided.
# If so, echo the inputs onto the command line such that they are present in the
# log output:

if [ "$#" -lt "$nxArg" ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ "$#" -gt "$nxArg" ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  echo "Input Arguments:"
  echo "#########################################################################"
  echo "Number of arguments.: $#"
  echo "List of arguments...: $@"
  echo "Arg #1..............: $1 (Input File)"
  inFile=$1
  #videoFile=$1
  echo "Arg #2..............: $2 (Output Parent Directory)"
  OutDir=$2
  echo "Arg #3..............: $3 (Output Sub-Directory)"
  OutDir_sub=$3
  echo "#########################################################################"
  echo ""
fi


# PARAMETERS
# ====================================================
# get the full path to the main package directory for this package on this machine
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PyanWhspEnvDir="$BASEDIR/envs/pyannote"
# ====================================================


echo "Current info on GPU from nvidia-smi:"
echo "===================================================="
nvidia-smi
echo "===================================================="
echo ""
## ====================================================

# RUN PYANNOTE/WHISPER SCRIPT
# ====================================================
# Go to the directory containing the venv...
cd $PyanWhspEnvDir

# Activate venv..
echo "activating Pyannote/Whisper venv ..."
echo ""
source env/bin/activate

# Enter venv env dir ..
cd env

# Echo message about call about to be run on the command line...
echo "Beginning the Pyannote/Whisper Pipeline to Make Diarized Whisper Transcripts:"
echo "======================================================="
# Run Pyannote/Whisper script.. 
#srun --nodes=1 --ntasks=1 python ./lib/python3.9/site-packages/wsprPyannoteTest1.py
#python ./lib/python3.9/site-packages/wsprPyannoteTest1.py
python ./lib/python3.8/site-packages/wsprPyannoteTest2.py $inFile $OutDir $OutDir_sub

echo "======================================================="
echo "Pyannote/Whisper Pipeline Completed ..."
echo ""

# Deactivate Venv
echo "deactivating Pyannote/Whisper venv ..."
echo ""
deactivate
# ====================================================
#/scratch/g/tark/dataScraping/envs/pyannote/env/lib/python3.9/site-packages/wsprPyannoteTest1.py
