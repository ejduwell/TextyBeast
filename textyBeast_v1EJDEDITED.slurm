#!/bin/bash

# data scraping whisper and mmocr combined pipeline.. v2

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=01:25:00
#SBATCH --account=tark
##SBATCH --qos=dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/g/tark/installTesting/dataScraping/output/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu

# Spit out job info..
echo "Starting at $(date)"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
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
nxArg=2

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
  echo "#########################################################################"
fi
# ====================================================


# PARAMETERS
# ====================================================
# General Paramters:
homeDir=/scratch/g/tark/installTesting/dataScraping #!!!!!!!!!!                                          
outBase=/scratch/g/tark/installTesting/dataScraping/output #!!!!!!!!!!
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start 
#strip away both path and extension from input video file to get base name
fBase=${videoFile%.*}
fBase=${fBase##*/}
outDir=(/scratch/g/tark/installTesting/dataScraping/output/$fBase-output-$dateTime) #make unique output directory name/path tagged with date and time #!!!!!!!!!!
imDirTmp=/scratch/g/tark/installTesting/dataScraping/tmp/images/ #!!!!!!!!!!
audDirTmp=/scratch/g/tark/installTesting/dataScraping/tmp/audio/ #!!!!!!!!!!
#videoFile=/scratch/g/tark/installTesting/dataScraping/video_test/test4_trimmed.mp4 #!!!!!!!!!!


# MMOCR Specific Parameters:
MMOCREnvDir=/scratch/g/tark/installTesting/dataScraping/envs/ocr #!!!!!!!!!!
vid_dir="/scratch/g/tark/installTesting/dataScraping/video_test/" #!!!!!!!!!!
video="$(basename $videoFile)" #strip path from video file to make input for mmocr..
frame_dsrate=5 # Specifies the frame rate at which video is initially sampled
cor_thr=0.95 # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"

#detector='PANet_IC15'
#detector='DBNet'
detector='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/panet_resnet18_fpem-ffm_600e_icdar2015.py'
#recognizer='SAR'
recognizer='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py'

x_merge=65
ClustThr_factor=3
det_ckpt_in='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth' #!!!!!!!!!!
recog_ckpt_in='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth' #!!!!!!!!!!

#det_ckpt_in='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth' #!!!!!!!!!!
#recog_ckpt_in='/scratch/g/tark/installTesting/dataScraping/envs/ocr/env/mmocrChkpts/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real_20220915_171910-04eb4e75.pth' #!!!!!!!!!!
mmocrOut_dir=$outDir/out_dir/video_img_dta

# Whisper Specific Parameters:
WhspEnvDir=/scratch/g/tark/installTesting/dataScraping/envs/whspr #!!!!!!!!!!
whspModel=large

#strip away both path and extension from input video file to get base name
audioBase=${videoFile%.*}
audioBase=${audioBase##*/}
#use base name above to construct full path name for eventual audio file created later in script..
audioFile=$outDir/out_dir/audio_speech_dta/$audioBase.wav

whsprOut_dir=$outDir/out_dir/audio_speech_dta
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
#mv $videoFile $outDir/$video
cp $videoFile $outDir/$video
vid_dir=$outDir #update the video directory to correspond to the output directory
videoFile=$outDir/$video

# Go there and make ocr output subdir
#cd $outDir
#mkdir ocrOut
#cd ocrOut #enter it
#ocrOut_dir=$(pwd) #save path location
#echo "Created output subdirectory for whisper at: $ocrOut_dir "

#cd $outDir
#mkdir whsprOut
#cd whsprOut #enter it
#whsprOut_dir=$(pwd) #save path location
#echo "Created output subdirectory for whisper at: $whsprOut_dir "
# ====================================================

echo "#########################################################################"
echo "#########################################################################"
echo "###########       EXTRACT TEXT FROM IMAGES WITH MMOCR         ###########"
echo "#########################################################################"
echo "#########################################################################"

# LOAD RCC CLUSTER MODULES
# ====================================================
module load python/3.9.1
module load ffmpeg
module load cuda/11.7.0
#module unload cudnn

# List currently loaded modules ..
module list

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
python ./lib/python3.8/site-packages/lecxr_text_v3.py $vid_dir $video $frame_dsrate $cor_thr $detector $recognizer $x_merge $ClustThr_factor $det_ckpt_in $recog_ckpt_in
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

# LOAD RCC CLUSTER MODULES
# ====================================================
#module load python/3.9.1 #ejd commented as this is already loaded earlier
#module load ffmpeg #ejd commented as this is already loaded earlier
#module unload cuda/11.7.0 #unload the cuda module loaded earlier..

# List currently loaded modules ..
module list

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
python ./WhsprOcrCombine.py $whsprFile $mmocrFile $outDir

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
mv $outBase/$fBase-output-$dateTime.tar.gz $outBase/$finSignal/output-$dateTime.tar.gz #move final zipped data into "end-signal" directory..
cp /scratch/g/tark/installTesting/dataScraping/output/*${SLURM_JOB_ID}.out $outBase/$finSignal/dtaScrape_${SLURM_JOB_ID}.out #move log file into "end-signal" directory..
rm -rf $outDir #get rid of original/unzipped data dir..

#Enter final output dir and give the signal that we're all done...
cd $outBase/$finSignal/ #enter final output dir..
sig="done"
echo $sig > DONE
# Go back to start dir..
cd $strtDir
rm /scratch/g/tark/installTesting/dataScraping/output/*${SLURM_JOB_ID}.out #clean up..