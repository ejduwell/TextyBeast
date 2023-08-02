#!/bin/bash

#data download script

# SLURM JOB INFO PARAMETERS
# ====================================================
#SBATCH --job-name=dtaScrape
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=00:20:00
#SBATCH --account=tark
#SBATCH --output=/scratch/g/tark/dataScraping/output/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eduwell@mcw.edu

# Spit out job info..
echo "Starting at $(date)"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "Transfering the outputs from job $6 to your local machine .."
echo ""
# ====================================================

# PROCESS INPUT ARGUMENTS
# ====================================================
# number of expected input arguments (must update if more are added in development):
nxArg=7

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
  #echo "Arg #1..............: $1 (Data Folder)"
  dtaFolder=$1
  #echo "Arg #2..............: (local username)"
  unLocal=$2
  #echo "Arg #3..............: (local pword)"
  pwLocal=$3
  #echo "Arg #4..............: (local ip)"
  ipLocal=$4
  #echo "Arg #5..............: (local outdir)"
  dirLocal=$5
  #echo "Arg #5..............: (original job id)"
  idOrig=$6
  finSignal=$7
  echo "#########################################################################"
fi
# ====================================================


# SEND THE DATA FOLDER
# ====================================================
# first add user/bin to path to get access to necessary bash functions..
export PATH="/scratch/g/tark/dataScraping/bin/sshpass/1.09/bin:$PATH"

#Enter /usr/bin/ and run ls to show whats inside...
dir1=$(pwd) #save start location
cd /usr/bin/
ls
cd $dir1 # go back to start location

# now use sshpass/scp to export the data back to the local machine..
/scratch/g/tark/dataScraping/bin/sshpass/1.09/bin/sshpass -p $pwLocal /usr/bin/scp -r  $outDir.tar.gz $unLocal@$ipLocal:$dirLocal
# ====================================================