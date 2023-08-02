#!/bin/bash

# for issueing calls to the whisper environment from other virtual environments..

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=02:50:00
#SBATCH --account=tark
##SBATCH --qos=dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/g/tark/installTesting/dataScraping/output/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu


# LOAD RCC CLUSTER MODULES
# ====================================================
module load python/3.9.1
module load ffmpeg
#module load cuda/11.7.0
module list

#SET UP WHISPER VIRTUAL PYTHON ENVIRONMENT
# ====================================================
source /scratch/g/tark/installTesting/dataScraping/envs/whspr/env/bin/activate


# PROCESS INPUT ARGUMENTS
# ====================================================
model=$1
dirOut=$2
lang=$3
InDir=$4
finSignal=$5

# Audio Files will be everything from input 6-->Last input..
# shift to get rid of first 5 arguments
shift 5
# put the remaining arguments in an array
file_list=("$@")

# RUN WHISPER
# ====================================================
# Loop over the files and run whisper command on each
for file in "${file_list[@]}"
do
  echo "Processing file: $file"
  echo "whisper $file --model $model --output_dir $dirOut --language $lang"
  whisper $file --model $model --output_dir $dirOut --language $lang
  
  # Replace this command with the one you want to run on each file
  #echo "whisper_bash $file $model $dirOut $lang $InDir"
  #whisper_bash $file $model $dirOut $lang $InDir
done

#DEACTIVATE WHISPER VIRTUAL PYTHON ENVIRONMENT
# ====================================================
deactivate

#CREATE THE FINISHED SIGNAL
# ====================================================
sig="done"
echo $sig > $finSignal
