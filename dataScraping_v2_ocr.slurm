#!/bin/bash

# ocrversion

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=07:20:00
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
# ====================================================

# PARAMETERS
# ====================================================
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
MMOCREnvDir=/scratch/g/tark/dataScraping/envs/ocr
outDir=(/scratch/g/tark/dataScraping/output/output-$dateTime) #make unique output directory name/path tagged with date and time
imDirTmp=/scratch/g/tark/dataScraping/tmp/images/
audDirTmp=/scratch/g/tark/dataScraping/tmp/audio/
audioFile=/scratch/g/tark/dataScraping/input/GMT20230207-180305_Recording.m4a #*may become input param..
#whspModel=large
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
mkdir ocrOut
cd ocrOut #enter it
ocrOut_dir=$(pwd) #save path location
echo "Created output subdirectory for whisper at: $ocrOut_dir "
# ====================================================

# LOAD RCC CLUSTER MODULES
# ====================================================
module load python/3.9.1
module load ffmpeg
module load cuda/11.7.0
module unload cudnn

# List currently loaded modules ..
module list
# ====================================================

# RUN MMOCR
# ====================================================

# Go to the directory containing the venv...
cd $MMOCREnvDir

# Activate venv..
source env/bin/activate

# Enter venv env dir ..
cd env

# Echo message about call about to be run on the command line...
echo "running the test python mmocr file.."


# Run MMOCR .py file .. 
# python ./testmmocr.py
python ./lib/python3.9/site-packages/lecxr_text_v3.py
echo "======================================================="
echo "Done..."
echo ""

# Deactivate Venv
deactivate
# ====================================================

# Go back to start dir..
cd $strtDir

# Close Up...
echo "Ending at $(date)"
