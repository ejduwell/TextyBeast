#!/bin/bash

# data scraping annotated interview script

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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu


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
  homeDir=$3
  echo "#########################################################################"
fi
# ====================================================

# PARAMETERS
# ====================================================
#-------------------------------------------------------------------------------
# General Paramters:
#-------------------------------------------------------------------------------
outBase=$homeDir/output
dateTime=$(date +%m-%d-%y-%H-%M-%S-%N) #get date and time of start
#strip away both path and extension from input video file to get base name
fBase=${videoFile%.*}
fBase=${fBase##*/}
outDir=($homeDir/output/$fBase-output-$dateTime) #make unique output directory name/path tagged with date and time
outDirBase=($homeDir/output)
outDirSub=$fBase-output-$dateTime
PyanWhspEnvDir="$homeDir/envs/pyannote"
videoFile=$homeDir/input/$videoFile
video="$(basename $videoFile)"
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
videoFile=$outDir/$video
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
python ./lib/python3.8/site-packages/wsprPyannoteTest2.py $videoFile $outDirBase $outDirSub

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
cp /scratch/g/tark/installTesting/dataScraping/output/$logfile $outBase/$finSignal/$dateTime-dtaScrape_$logfile #move log file into "end-signal" directory..
rm -rf $outDir #get rid of original/unzipped data dir..

#Enter final output dir and give the signal that we're all done...
cd $outBase/$finSignal/ #enter final output dir..
sig="done"
echo $sig > DONE
# Go back to start dir..
cd $strtDir
