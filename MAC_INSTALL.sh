#!/bin/bash

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
echo "######################### INSTALLER FOR MAC #############################"
echo "#########################################################################"
echo ""
echo ""

echo "             Welcome. This is the installer script for Mac.              "
echo "                     Continue with the installation?:                    "
echo "    Press 'y' below to proceed or 'n' to exit. The press Enter/Return    "
echo ""
read -p "                          Proceed?: " prcd1
echo ""

# continue only if prcd='y'
#--------------------------------------
if [[ $prcd1 == 'y' ]]; then

echo "Continuing with installation ..."
echo ""
echo "########################################"
echo "# Building Virtual Python Environments #" 
echo "########################################"
echo ""

# make necessary scripts executable..
chmod +wrx makeExctbl; ./makeExctbl

# Build Virtual Python Environments
./buildEnvs_Mac.sh

echo ""
echo "########################################"
echo "#       Check Local Installation       #" 
echo "########################################"
echo ""
echo "The following routine opens a couple    "
echo "jupyter notebooks containing some code  "
echo "to test your local install is working.  "
echo ""
read -p "   Press 'y' run or 'n' to skip: " prcd2

if [[ $prcd2 == 'y' ]]; then
./checkInstall.sh
fi

# Enter TextyBeast/textyBeastLocal
# and make all scripts executable ..
cd TextyBeast/textyBeastLocal
chmod +wrx *

echo ""
echo "########################################"
echo "#     Remote System Setup/Install      #" 
echo "########################################"
echo ""
echo "Do you want to setup the capability to  "
echo "submit jobs to a remote SLURM Cluster   "
echo "or some other remote machine?           "

echo ""
read -p "Press 'y' run remote setup or 'n' to skip: " prcd3
echo ""

if [[ $prcd3 == 'y' ]]; then

# Run setup.sh script:
./setup.sh

#source the shell profile script
echo ""
echo "Now we need to source your shell profile"
echo "script to make the info you just entered"
echo "available withing this session ..."
echo ""
echo "Is your primary shell zsh of bash?:     "
echo "----------------------------------------"
echo "(If you're  not sure, open a new        "
echo "terminal window and check the prompt:   "
echo "zsh will have a '%', bash have a '$')   "
echo "----------------------------------------"
read -p "Press 'b' for bash or 'z' for zsh: " shellResp
echo ""

if [[ $shellResp == 'b' ]]; then
source ~/.bash_profile
fi

if [[ $shellResp == 'z' ]]; then
source ~/.zshrc
fi

echo ""
echo "Installing TextyBeast on Remote Cluster:"
echo "========================================"
echo ""
./remoteClstrInstall.sh
echo ""
echo "========================================"
echo ""

echo "****************************************"
echo "***************** NOTE *****************"
echo "****************************************"
echo ""
echo "If you want to submit jobs to a SLURM   "
echo "cluster, there is one final thing you   "
echo "to do after completing this script.     "
echo ""
echo "You will need to make some small edits  "
echo "to the SLURM headers of two files on the"
echo "cluster to configure them for your SLURM"
echo "account. Instructions are pasted below: "
echo ""
echo "1) First ssh into the cluster:           "
echo "-----------------------------------------"
echo "ssh <yourUsername>@<yourSlurmClusterHostname/IP>"
echo "-----------------------------------------"
echo ""
echo "2) Navigate to the main TextyBeast directory:"
echo "-----------------------------------------"
echo "cd /path/to/TextyBeast"
echo "-----------------------------------------"
echo ""
echo "3) Use your favorite text editor to edit "
echo "   the following two files:"
echo "    - textyBeast_v1_slurm.sh             "
echo "    - pyAnWhsp_v1_slurm.sh               "
echo "-----------------------------------------"
echo "      - - - - - - Example - - - - -      "
echo ""
echo "nano textyBeast_v1_slurm.sh"
echo ""
echo "At the very top of each file under a commented"
echo "header that reads 'SLURM JOB INFO PARAMETERS' "
echo "you should see a list of options all listed   "
echo "after '#SBATCH:'"
echo ""
echo "# SLURM JOB INFO PARAMETERS"
echo "# ===================================================="
echo "#SBATCH --job-name=dtaScrape"
echo "#SBATCH --ntasks=1"
echo "#SBATCH --mem-per-cpu=10gb"
echo "#SBATCH --time=01:25:00"
echo "#SBATCH --account=tark"
echo "##SBATCH --qos=dev"
echo "#SBATCH --partition=gpu"
echo "#SBATCH --gres=gpu:1"
echo "#SBATCH --mail-type=ALL"
echo "#SBATCH --mail-user=eduwell@mcw.edu"
echo "# ===================================================="
echo ""
echo " - set the --account option equal to your account name"
echo ""
echo " - set the --mail-type option equal to ALL and the      "
echo "  --mail-user option equal to your email address.       "
echo "  This will tell the cluster to send you an email when  "
echo "  your job starts and finishes. (alternatively, you can "
echo "  comment/delete this line if you don't want this..)    "
echo ""
echo " - save your changes, and you should be good to go ...  "
echo ""
echo "      - - - - - - End Example - - - - -      "
echo ""
echo ""
echo "-----------------------------------------"
echo ""
echo "****************************************"
echo "*** NOTE: FOLLOW INSTRUCTIONS ABOVE  ***"
echo "******* TO COMPLETE SLURM SETUP  *******"
echo "****************************************"
echo ""
echo "Follow instructions above to complete   "
echo "SLURM setup after this script finishes.."
echo ""
read -p "Press any key to continue ..." anyKey
echo ""
fi


echo ""
echo "########################################"
echo "#   Add Functions to Permenant Path    #" 
echo "########################################"
echo ""
echo "Finally.. Adding TextyBeast functions   "
echo "to the permenant local path ...         "
echo ""
read -p "Press any key to continue ..." anyKey
echo ""

# Add fucntions in textyBeastLocal to permenant path
./addFcns2Path.sh

echo ""
echo "TextyBeast Install Complete!"
echo ""
fi
#--------------------------------------
