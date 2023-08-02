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
echo "################### REMOTE CLUSTER INSTALLATION #########################"
echo "#########################################################################"
echo ""



echo ""
echo "Fetching Remote Cluster Info Provided During Install.."
echo "#########################################################################"
# Get cluster location info for cluster
# (from .bashrc/.bash_profile ...)
# (clstrPath and defined when running setup.sh ...)
remoteDir=$clstrPath
#remoteDir="/scratch/g/tark/installTesting/dataScraping"
echo ""
echo "Remote Cluster Hostname:"
echo $clstrHost
echo ""
echo "Remote Cluster Install Location:"
echo $remoteDir
echo ""
echo "Remote Cluster Username:"
uservar=$clstrUsr
echo $clstrUsr
echo ""
echo "#########################################################################"
echo ""
echo ""

installCmdLst="cd $clstrPath; git clone https://github.com/ejduwell/TextyBeast.git; chmod +wrx install.sh; ./install.sh"

ssh $uservar@$clstrHost $installCmdLst
