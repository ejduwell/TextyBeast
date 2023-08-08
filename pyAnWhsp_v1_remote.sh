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
nxArg=6

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
  #echo "List of arguments...: $@"
  #echo "Arg #1..............: $1 (Video File)"
  videoFile=$1
  
  #echo "Arg #2..............: (local username)"
  #unLocal=$2
  #echo "Arg #3..............: (local pword)"
  #pwLocal=$3
  #echo "Arg #4..............: (local ip)"
  #ipLocal=$4
  #echo "Arg #5..............: (local outdir)"
  #dirLocal=$5
  
  finSignal=$2
  tokenIn=$3
  outDirFinal=$4
  parsFile=$5
  homeDir=$6
  echo "#########################################################################"
  echo ""
fi


# PARAMETERS
# ====================================================
#-------------------------------------------------------------------------------
# General Paramters:
#-------------------------------------------------------------------------------
# get the full path to the main package directory for this package on this machine
BASEDIR=$homeDir
#BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#homeDir=$BASEDIR;

outBase=$homeDir/output
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
#strip away both path and extension from input video file to get base name
fBase=${videoFile%.*}
fBase=${fBase##*/}
outDir=($homeDir/output/$fBase-output-$dateTime) #make unique output directory name/path tagged with date and time
outDirBase=($homeDir/output)
outDirSub=$fBase-output-$dateTime
PyanWhspEnvDir="$homeDir/envs/pyannote"
video="$(basename $videoFile)"
# ====================================================
#-------------------------------------------------------------------------------
# SOURCE PARS FILE
#-------------------------------------------------------------------------------

if [[ $parsFile == "default" ]]; then
echo ""
echo "Sourcing Default Parameters:"
curDir=$(pwd)
cd $BASEDIR
source defaultPars.sh
cd $curDir

else
echo ""
echo "Sourcing Parameters from Input Pars.sh File:"
curDir=$(pwd)
cd $BASEDIR
source $parsFile
cd $curDir
fi

echo "frame_dsrate: $frame_dsrate"
echo "cor_thr: $cor_thr"
echo "detector: $detector"
echo "recognizer: $recognizer"
echo "Detector: $x_merge"
echo "x_merge: $ClustThr_factor"
echo "det_ckpt_in: $det_ckpt_in"
echo "recog_ckpt_in: $recog_ckpt_in"
echo "whspModel: $whspModel"
#-------------------------------------------------------------------------------


# SET UP PATH/DIRECTORIES
# ====================================================
# Save start directory...
strtDir=$(pwd)

# Make output directory for this job
mkdir $outDir
echo "Created output directory for this job at:" 
echo "$outDir"
echo ""

# Copy video file into output directory
cp $videoFile $outDir/$video
videoFile=$outDir/$video
# ====================================================

module load ffmpeg

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
python ./lib/python3.8/site-packages/wsprPyannote2.py $videoFile $outDirBase $outDirSub $tokenIn $BASEDIR $whspModel

echo "======================================================="
echo "Pyannote/Whisper Pipeline Completed ..."
echo ""

# Deactivate Venv
echo "deactivating Pyannote/Whisper venv ..."
echo ""
deactivate
# ====================================================


# Go back to start dir..
cd $strtDir
# Close Up...
echo "Ending at $(date)"

# PACK/CLEAN UP OUTPUT DIRS
# ====================================================
# first compress the output dir ..
cd $outBase
tar -czvf $fBase-output-$dateTime.tar.gz $fBase-output-$dateTime
#tar -czvf $outDir.tar.gz $outDir


mkdir $outBase/$finSignal
#mv  $outDir.tar.gz $outBase/$finSignal/output-$dateTime.tar.gz #move final zipped data into "end-signal" directory..
mv $outBase/$fBase-output-$dateTime.tar.gz $outBase/$finSignal/output-$dateTime.tar.gz #move final zipped data into "end-signal" directory..

logfile=$(basename "$videoFile").out
cp $BASEDIR/output/$logfile $outBase/$finSignal/$dateTime-dtaScrape_$logfile #move log file into "end-signal" directory..
rm -rf $outDir #get rid of original/unzipped data dir..

#Enter final output dir and give the signal that we're all done...
cd $outBase/$finSignal/ #enter final output dir..

sig="done"
echo $sig > DONE
# Go back to start dir..
cd $strtDir

# Clean up output directory
cd $BASEDIR/output/
rm $logfile

# Clean up input directory
cd $BASEDIR/input
rm $video # remove video from input

cd $strtDir
