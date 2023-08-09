#!/bin/bash



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
echo ""
echo "#########################################################################"
echo "#########################################################################"
echo "##                   -------------------------------                   ##"
echo "##                   Remote Location/Path/SSH Setup:                   ##"
echo "##                   -------------------------------                   ##"
echo "##      Welcome! This script is for specifying info about your         ##"
echo "##      SLURM computing cluster which will be used in establishing     ##"
echo "##      connections to submit jobs from your local machine.            ##"
echo "##                                                                     ##"
echo "##      This script must be run as part of your install process        ##"
echo "##      before you are able to submit jobs to a cluster                ##"
echo "##      or remote machine.                                             ##"
echo "##                                                                     ##"
echo "##      However, you should only need to run this once during          ##"
echo "##      the install process/won't need to do this again.               ##"
echo "##      The info you enter here will be saved in your .bashrc          ##"
echo "##      or .bash_profile script and will automatically                 ##"
echo "##      be available to your system after completing the               ##"
echo "##      following prompts.                                             ##"
echo "##                                                                     ##"
echo "#########################################################################"
echo "#########################################################################"
echo ""
echo ""

# READ IN INPUTS FROM USER
echo "Is your local machine running MacOSX or a Linux distro?"
echo "Press 'm' for Mac or 'l' for Linux. Then press enter/return:"
read osType
echo ""

echo ""
echo "Please enter the hostname or IP of the SLURM cluster host below:"
read clstrHost
echo ""

echo ""
echo "Please enter the full path to the parent folder on the SLURM cluster where you want to install TextyBeast below:"
read clstrPath
echo ""

# ADD THESE TO THE PERMANENT PATH BY ADDING CALLS TO .BASHRC/.BASHPROFILE
#----------------------------------------------------------------------------------

# First check whether they indicated mac or linux and assign rcFile to the relavent
# .bashrc/.bash_profile string..

if [[ $osType == "l" ]]; then
rcFile=~/.bashrc
fi

if [[ $osType == "m" ]]; then
rcFile=~/.bash_profile
fi

echo "Adding cluster path and hostname info to permanent path"
echo "by adding the following commands to the end of $rcFile:"
echo "------------------------------------------------------------"
# First Add Comment In .bashrc/.bash_profile..
echo "">> $rcFile
echo "# Export Variables Indicating TextyBeast Remote Cluster Info">> $rcFile
echo "# ---------------------------------------------------------">> $rcFile


# build and add export command to export clstrHost from bashrc file:
Cmd1_1='export clstrHost='
Cmd1=$Cmd1_1'"'$clstrHost'"'
echo $Cmd1 >> $rcFile
echo $Cmd1

# build and add export command to export clstrPath from bashrc file:
Cmd2_1='export clstrPath='
Cmd2=$Cmd2_1'"'$clstrPath'"'
echo $Cmd2 >> $rcFile
echo $Cmd2

# build and add export command to export rcFile from bashrc file:
# (this is so other textyBeast programs can know name/path of the
# bashrc/bash_profile initialization file...
Cmd3_1='export rcFile='
Cmd3=$Cmd3_1'"'$rcFile'"'
echo $Cmd3 >> $rcFile
echo $Cmd3
echo "------------------------------------------------------------"



echo ""
echo ""
echo "SSH KEY SETUP:"
echo "#########################################################################"
echo ""
echo "Please enter your username on the cluster below:"
read -p 'Username: ' uservar
echo ""
echo "If you have not done so already, you will need to set up a public/private"
echo "SSH key on your local machine.. this allows for automation of sending commands"
echo "to the cluster via SSH using password protected keys rather than manual"
echo "password entry."
echo ""
echo "Without setting this up, you will need to enter login credentials every single"
echo "time a command is issued to the cluster via SSH, which will make running jobs"
echo "very cumbersome/unworkable..."
echo ""
echo "If you say yes below and run the setup procedure, you will be asked"
echo "to enter the password you establish to protect the ssh keys at the beginning"
echo "of each bash session. This will then allow you to submit ssh calls to the the"
echo "cluster or any other location with ssh keys in your .ssh/id_rsa file without"
echo "requiring a password again within that terminal session."
echo ""
echo "Would you like to set that up now? (HIGHLY RECOMMENDED)"
read -p '(enter either y or n, then hit ENTER) ' sshkeySetup

STR=$sshkeySetup
SUB='y'
SUB2='n'
SUB3='yn'

if [[ "$STR" == "$SUB" ]]; then
    echo "OK.. setting up SSH keys.."
    echo ""
    echo "Please follow the instructions to complete SSH key setup:"
    echo "---------------------------------------------------------"
 
    #ssh-keygen -t rsa -b 2048
    ssh-keygen -p -f ~/.ssh/id_rsa
    eval "$(ssh-agent -s)"
    ssh-copy-id $uservar@$clstrHost
    # add to your shell profile file
    echo 'eval "$(ssh-agent -s)"' >> $rcFile
    echo 'ssh-add ~/.ssh/id_rsa' >> $rcFile
    echo ""
    echo "---------------------------------------------------------"
    echo ""
    echo "Copying your keys to the Cluster..."

elif [[ "$STR" == "$SUB2" ]]; then
    echo "OK, proceeding without SSH key setup.."
else
    echo "input not recognized.. try again"
    read -p '(enter either y or n, then hit ENTER) ' sshkeySetup
    while [[ "$SUB3" != *"$sshkeySetup"* ]]
    do
	   echo "input not recognized.. try again"
	   read -p '(enter either y or n, then hit ENTER) ' sshkeySetup
    done

    STR=$sshkeySetup
    if [[ "$STR" == "$SUB" ]]; then
	 echo "OK.. setting up SSH keys.."
	 echo ""
	 echo "Please follow the instructions to complete SSH key setup:"
	 echo "---------------------------------------------------------"
	 
	 #ssh-keygen -t rsa -b 2048
	 ssh-keygen -p -f ~/.ssh/id_rsa
	 eval "$(ssh-agent -s)"
  	 ssh-copy-id $uservar@$clstrHost
	 # add to your shell profile file
	 echo 'eval "$(ssh-agent -s)"' >> $rcFile
	 echo 'ssh-add ~/.ssh/id_rsa' >> $rcFile
	 echo ""
	 echo "---------------------------------------------------------"
	 
	 echo ""
	 echo "Copying your keys to the Cluster..."

    elif [[ "$STR" == "$SUB2" ]]; then
	echo "OK, proceeding without SSH key setup.."
    fi
fi


# Finally, add calls to export the cluster username variable via .bashrc
Cmd4_1='export clstrUsr='
Cmd4=$Cmd4_1'"'$uservar'"'
echo $Cmd4 >> $rcFile
echo ""

echo "Adding cluster username info to permanent path by adding"
echo "the following command to the end of $rcFile:"
echo "---------------------------------------------------------"
echo $Cmd4
echo "---------------------------------------------------------"

echo ""
echo ""
echo "Thank you, $uservar, SSH key setup is complete..."
echo ""
echo "#########################################################################"
echo ""
echo ""

# add final line to cordon off this section of the .bashrc/.bash_profile
echo "# ---------------------------------------------------------">> $rcFile

echo ""
echo "This concludes cluster/location path setup.."
echo "To make the variables exported above available"
echo "within this terminal session, run:"
echo "source $rcFile"
echo ""
echo "Otherwise, in the future, these variables will"
echo "be sourced at the onset of each new terminal"
echo "bash session when "$rcFile" is sourced automatically..."
echo ""
echo ""
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -NOTE- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "If, for whatever reason, you need to run this script again"
echo "in the future:"
echo "------------------------------------------------------------------------"
echo "Make sure you first go to your $rcFile, and clear"
echo "out the commands entered during this session.. They will be listed under:"
echo "'" "# Export Variables Indicating TextyBeast Remote Cluster Info" "'"
echo ""
echo "------------------------------------------------------------------------"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo ""
echo "Whelp.. See ya later..."
echo ""
