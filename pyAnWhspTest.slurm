#!/bin/bash

# data scraping whisper and mmocr combined pipeline.. v2

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --nodes=1
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=01:00:00
#SBATCH --account=tark
##SBATCH --qos=dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/g/tark/installTesting/dataScraping/output/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu



# PARAMETERS
# ====================================================
PyanWhspEnvDir="/scratch/g/tark/installTesting/dataScraping/envs/pyannote"
# ====================================================

# LOAD RCC CLUSTER MODULES
# ====================================================
module load python/3.9.1
module load ffmpeg
module load RCC

# List currently loaded modules ..
module list

#echo "Current info on GPU from nvidia-smi:"
#echo "===================================================="
#nvidia-smi
#echo "===================================================="
#echo ""
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
python ./lib/python3.8/site-packages/wsprPyannoteTest2.py

echo "======================================================="
echo "Pyannote/Whisper Pipeline Completed ..."
echo ""

# Deactivate Venv
echo "deactivating Pyannote/Whisper venv ..."
echo ""
deactivate
# ====================================================
#/scratch/g/tark/dataScraping/envs/pyannote/env/lib/python3.9/site-packages/wsprPyannoteTest1.py