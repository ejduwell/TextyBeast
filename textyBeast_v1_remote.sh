#!/bin/bash

# data scraping whisper and mmocr combined pipeline.. locally..

# Spit out job info..
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
echo ""
# ====================================================

# PROCESS INPUT ARGUMENTS
# ====================================================
# number of expected input arguments (must update if more are added in development):
nxArg=5

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
  outDirFinal=$3
  parsFile=$4
  homeDir=$5
  echo "#########################################################################"
fi
# ====================================================


# PARAMETERS
# ====================================================
#-------------------------------------------------------------------------------
# General Paramters:
#-------------------------------------------------------------------------------
BASEDIR=$homeDir
# get the full path to the main package directory for this package on this machine

#BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#homeDir=$BASEDIR;

outBase=$homeDir/output
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
#strip away both path and extension from input video file to get base name
fBase=${videoFile%.*}
fBase=${fBase##*/}
outDir=($homeDir/output/$fBase-output-$dateTime) #make unique output directory name/path tagged with date and time

#-------------------------------------------------------------------------------
# MMOCR Specific Parameters:
#-------------------------------------------------------------------------------
MMOCREnvDir=$homeDir/envs/ocr
video="$(basename $videoFile)" #strip path from video file to make input for mmocr..
mmocrOut_dir=$outDir/out_dir/video_img_dta
configsDir=$homeDir/"envs/ocr/env/lib/python3.8/site-packages/mmocr/configs"

#frame_dsrate=5 # Specifies the frame rate at which video is initially sampled
#cor_thr=0.95 # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
#detector='PANet_IC15'
#recognizer='SAR'
#x_merge=65
#ClustThr_factor=3
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth'



#-------------------------------------------------------------------------------
# Whisper Specific Parameters:
#-------------------------------------------------------------------------------
#whspModel=base

WhspEnvDir=$homeDir/envs/whspr
#strip away both path and extension from input video file to get base name
audioBase=${videoFile%.*}
audioBase=${audioBase##*/}
#use base name above to construct full path name for eventual audio file created later in script..
audioFile=$outDir/out_dir/audio_speech_dta/$audioBase.wav
whsprOut_dir=$outDir/out_dir/audio_speech_dta

#-------------------------------------------------------------------------------
# SOURCE PARS FILE
#-------------------------------------------------------------------------------

if [[ $parsFile == "default" ]]; then
echo ""
echo "Sourcing Default Parameters:"
#curDir=$(pwd)
#cd $BASEDIR
source $homeDir/defaultPars.sh
#cd $curDir

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
# ====================================================

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
vid_dir=$outDir #update the video directory to correspond to the output directory
videoFile=$outDir/$video
# ====================================================

echo "#########################################################################"
echo "#########################################################################"
echo "###########       EXTRACT TEXT FROM IMAGES WITH MMOCR         ###########"
echo "#########################################################################"
echo "#########################################################################"

echo "Current info on GPU from nvidia-smi:"
echo "===================================================="
nvidia-smi
echo "===================================================="
echo ""
# ====================================================

# RUN MMOCR
# ====================================================
# Go to the directory containing the venv...
cd $MMOCREnvDir

# Activate venv..
echo "activating mmocr venv ..."
echo ""
source env/bin/activate

# Enter venv env dir ..
cd env

# Echo message about call about to be run on the command line...
echo "Beginning the MMOCR Pipeline to Detect Text in Image Frames:"
echo "======================================================="
# Run MMOCR pipeline script.. 
python ./lib/python3.8/site-packages/lecxr_text_v3.py $vid_dir $video $frame_dsrate $cor_thr $detector $recognizer $x_merge $ClustThr_factor $det_ckpt_in $recog_ckpt_in $configsDir
echo "======================================================="
echo "MMOCR Pipeline Completed ..."
echo ""

# Deactivate Venv
echo "deactivating mmocr venv ..."
echo ""
deactivate
# ====================================================

echo "#########################################################################"
echo "#########################################################################"
echo "###########       EXTRACT TEXT FROM AUDIO WITH WHISPER        ###########"
echo "#########################################################################"
echo "#########################################################################"

echo "Current info on GPU from nvidia-smi:"
echo "===================================================="
nvidia-smi
echo "===================================================="
echo ""
# ====================================================

# RUN WHISPER
# ====================================================
# Go to the directory containing the whisper venv...
cd $WhspEnvDir

# Activate whisper venv..
echo "activating whisper virtualenv ..."
echo ""
source env/bin/activate

# Echo whisper call about to be run on the command line...
echo "Recognizing and Transcribing Speech in $audioFile Using OpenAI's Whisper"
echo "(currently using the $whspModel model)" 
echo ""
echo "Bash Command Issued to Whisper:"
echo "whisper $audioFile --model $whspModel --output_dir $whsprOut_dir --language English"
echo ""
echo "Whisper Model Output:"
echo "======================================================="

# Run Whisper on Audio File
whisper $audioFile --model $whspModel --output_dir $whsprOut_dir --language English

# Rename output files such that there are not double extensions..
#cd $whsprOut_dir
mv $whsprOut_dir/$audioBase.wav.tsv $whsprOut_dir/$audioBase.tsv
mv $whsprOut_dir/$audioBase.wav.json $whsprOut_dir/$audioBase.json
mv $whsprOut_dir/$audioBase.wav.srt $whsprOut_dir/$audioBase.srt
mv $whsprOut_dir/$audioBase.wav.txt $whsprOut_dir/$audioBase.txt
mv $whsprOut_dir/$audioBase.wav.vtt $whsprOut_dir/$audioBase.vtt

echo "======================================================="
echo "Whisper Audio Transcription Complete ..."
echo ""

# Deactivate Whisper Venv
echo "deactivating whisper virtualenv ..."
echo ""
deactivate
# ====================================================

# AMALGAMATE THE WHISPER AND MMOCR TEXT DETECTION OUTPUTS INTO COMBINED OUTPUT FILE
whsprFile=$whsprOut_dir/$audioBase.tsv
mmocrTag="_ufTxt-Time"
mmocrFile=$mmocrOut_dir/$audioBase$mmocrTag.csv


# Go to the home directory and run the WhsprOcrCombine.py script..
cd $homeDir
source envs/ocr/env/bin/activate
python ./WhsprOcrCombine.py $whsprFile $mmocrFile $outDir
deactivate

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
