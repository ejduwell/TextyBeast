#!/bin/bash

# data scraping whisper and mmocr combined pipeline..

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=05:20:00
#SBATCH --account=tark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/g/tark/dataScraping/output/%x-%j.out
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

# PARAMETERS
# ====================================================
# General Paramters:
homeDir=/scratch/g/tark/dataScraping/
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
outDir=(/scratch/g/tark/dataScraping/output/output-$dateTime) #make unique output directory name/path tagged with date and time
imDirTmp=/scratch/g/tark/dataScraping/tmp/images/
audDirTmp=/scratch/g/tark/dataScraping/tmp/audio/
videoFile=/scratch/g/tark/dataScraping/video_test/CHA_I_Introduction_to_the_Nervous_System_default.mp4


# MMOCR Specific Parameters:
MMOCREnvDir=/scratch/g/tark/dataScraping/envs/ocr
vid_dir="/scratch/g/tark/dataScraping/video_test/"
video="$(basename $videoFile)" #strip path from video file to make input for mmocr..
frame_dsrate=15 # Specifies the frame rate at which video is initially sampled
cor_thr=0.95 # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
detector='PANet_IC15'
recognizer='SAR'
x_merge=65
ClustThr_factor=3
det_ckpt_in='/scratch/g/tark/dataScraping/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
recog_ckpt_in='/scratch/g/tark/dataScraping/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
mmocrOut_dir=$outDir/out_dir/video_img_dta

# Whisper Specific Parameters:
WhspEnvDir=/scratch/g/tark/dataScraping/envs/whspr
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
python ./lib/python3.9/site-packages/lecxr_text_v3.py $vid_dir $video $frame_dsrate $cor_thr $detector $recognizer $x_merge $ClustThr_factor $det_ckpt_in $recog_ckpt_in
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
echo "whisper $audioFile --model $whspModel --output_dir $whsprOut_dir"
echo ""
echo "Whisper Model Output:"
echo "======================================================="

# Run Whisper on Audio File
whisper $audioFile --model $whspModel --output_dir $whsprOut_dir

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