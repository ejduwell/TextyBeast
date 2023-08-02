#!/bin/bash

# Set some path variables first..
#===============================================================================
# get the full path to the main package directory for this package on this machine
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#===============================================================================



# Test the mmocr install by opening and running the jupyter notebook
#===============================================================================

# go the the ocr environment directory
cd $BASEDIR/envs/ocr/

# activate the ocr virtualenv
echo ""
echo "activating the mmocr virtual environment..."
source env/bin/activate

# now go to the mmocr demo directory and run the installation check notebook
cd $BASEDIR/envs/ocr/env/lib/python3.8/site-packages/mmocr/demo

echo ""
echo "Greetings! To test your mmocr installation we will now have you open and run some test code in a jupyter notebook..."
echo "If you press 'y' below, a jupyter notebook should open in your browser window with further instructions..."
echo ""
echo "Please press 'y' to continue or any other key to skip:"

read input1

if [[ $input1 == "y" ]]; then
	jupyter notebook TextyBeast_MMOCR_Install_Check.ipynb
fi

# deactivate the ocr virtualenv
echo ""
echo "deactivating the mmocr virtual environment..."
deactivate

# return to the base directory
cd $BASEDIR

#===============================================================================

echo ""
echo "exited mmocr install-test jupyter notebook..."
echo "please press ENTER/RETURN to proceed..."
read proceed

# Test the whisper install by opening and running the test audio
#================================================================================

# go the the whisper environment directory
cd $BASEDIR/envs/whspr/

# activate the whisper virtualenv
echo ""
echo "activating the whisper virtual environment..."
source env/bin/activate

# now go to the whisper installTest directory and run the installation check notebook
cd $BASEDIR/install/whspr/installTest

echo ""
echo "To test your whisper installation, we will, again, have you open and run some test code in a jupyter notebook..."
echo "If you press 'y' below, a jupyter notebook should open in your browser window with further instructions..."
echo ""
echo "Please press 'y' to continue or any other key to skip:"

read input2

if [[ $input2 == "y" ]]; then
        jupyter notebook TextyBeast_Whspr_Install_Check.ipynb
fi

# deactivate the whisper virtualenv
echo ""
echo "deactivating the mmocr virtual environment..."
deactivate

# return to the base directory
cd $BASEDIR

#===============================================================================

echo ""
echo "You've reached the end of the checkInstall script..."
echo "Take care now. Bye-Bye then."
echo ""
