#!/bin/bash

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=03:20:00
#SBATCH --account=tark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/g/tark/installTesting/dataScraping/output/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu

# Spit out job info..
echo "Starting at $(date)"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo ""
# ====================================================

# PARAMETERS
# ====================================================
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
WhspEnvDir=$BASEDIR/envs/whspr
outDir=($BASEDIR/output/output-$dateTime) #make unique output directory name/path tagged with date and time
imDirTmp=$BASEDIR/tmp/images/
audDirTmp=/scratch/g/tark/dataScraping/tmp/audio/
audioFile=/scratch/g/tark/dataScraping/input/GMT20230207-180305_Recording.m4a #*may become input param..
whspModel=large
# ====================================================

# SET UP PATH/DIRECTORIES
# ====================================================
# Save start directory...
strtDir=$(pwd)

# Make output directory for this job
mkdir $outDir
echo "Created output directory for this job at: $outDir "

# Go there and make whisper output subdir
cd $outDir
mkdir whsprOut
cd whsprOut #enter it
whsprOut_dir=$(pwd) #save path location
echo "Created output subdirectory for whisper at: $whsprOut_dir "
# ====================================================

# LOAD RCC CLUSTER MODULES
# ====================================================
module load python/3.9.1
module load ffmpeg
# ====================================================

# RUN WHISPER
# ====================================================

# Go to the directory containing the whisper venv...
cd $WhspEnvDir

# Activate whisper venv..
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

echo "======================================================="
echo "Transcription Done..."
echo ""

# Deactivate Whisper Venv
deactivate
# ====================================================

# Go back to start dir..
cd $strtDir

# Close Up...
echo "Ending at $(date)"
