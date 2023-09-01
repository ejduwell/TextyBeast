#!/bin/bash


# Set some path variables first..
#===============================================================================
# get the full path to the main package directory for this package on this machine
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#BASEDIR=$SCRAPINGBASEDIR

# Environment Setup
#===============================================================================
#create empty directories for envs, input, output
mkdir envs
mkdir input
mkdir output

# enter envs dir
cd envs

# list of environments
envs=("ocr" "whspr" "pyannote" "gui")

# For development: if you want to just install one or a subset of the envs to test,
# paste them in below, and uncomment it/comment the original above....
#envs=("gui") # temp line for expediency during development... remove later..

for env in ${envs[@]}; do
    # Save starting directory location
    # in this loop for later so we can return...
    strtDir=$(pwd)
    # create virtual environment
    mkdir $env
    cd $env
    if [[ $env == "ocr" ]]; then
    echo ""
    echo "------------------------------------------------------"
    echo "------- BUILDING VIRTUAL ENVIRONMENT FOR MMOCR -------"
    echo "------------------------------------------------------"
    echo ""
    virtualenv --python=python3.8 env
    fi

    if [[ $env == "pyannote" ]]; then
    echo ""
    echo "------------------------------------------------------"
    echo "----- BUILDING VIRTUAL ENVIRONMENT FOR PYANNOTE ------"
    echo "------------------------------------------------------"
    echo ""
    virtualenv --python=python3.8 env
    fi

    if [[ $env == "whspr" ]]; then
    echo ""
    echo "------------------------------------------------------"
    echo "------ BUILDING VIRTUAL ENVIRONMENT FOR WHISPER ------"
    echo "------------------------------------------------------"
    echo ""
    virtualenv --python=python3.9 env
    fi
    
    if [[ $env == "gui" ]]; then
    echo ""
    echo "------------------------------------------------------"
    echo "------ BUILDING VIRTUAL ENVIRONMENT FOR THE GUI ------"
    echo "------------------------------------------------------"
    echo ""
    virtualenv --python=python3.9 env
    fi

    # activate the virtual environment
    source env/bin/activate

    # upgrade pip
    pip install --upgrade pip

    # mmocr specific install stuff..
    # if this pass is the ocr env setup:
    # do the mmocr install procedure first
    # then move on to the requirements.txt
    if [[ $env == "ocr" ]]; then
	#First install the precise pytorch setup we need
	pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

	#Then run the requirements.txt
	pip install -r $BASEDIR/install/$env/requirements.txt

        #Next install openmim and mmengine (openmmlabs installer tools)
	pip install -U openmim
	mim install mmengine

	#Then use mim toinstall the precise versions of mmcv and mmdet used in our setup
	# (1.7.1 and 2.28.2 respectively)
	mim install mmcv==1.7.1
	mim install mmdet==2.28.2

	# Finally, install mmocr branch 0.x from our fork of the github repository
        cd $BASEDIR/envs/$env/env/lib/python3.8/site-packages
        git clone -b 0.x https://github.com/ejduwell/mmocr.git
        cd mmocr
        pip install -v -e .
	
	# Go to the mmocrChkpts dir and run installConfigs to pull local copies of models..
	cd $BASEDIR/install/$env/mmocrChkpts
	chmod +wrx installConfigs #ensure its executeable...
	./installConfigs #run it
	
        # install jupyter notebooks for running install checker notebook
        pip install jupyter
    else
        if [[ $env == "pyannote" ]]; then
            #pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
            pip install pyannote.audio==2.1.1
	    pip install pyannote.metrics==3.2.1
            pip install pyannote.core==4.5
            pip install pyannote.database==4.1.3
            pip install pyannote.pipeline==2.3
            pip install pydub
            pip install moviepy
        fi
        
        if [[ $env == "gui" ]]; then
            pip install tk
            pip install pillow
	    pip install pytube
	    pip install moviepy
	    pip install pygame
	    dirTmp=$(pwd)
	    cd $BASEDIR/envs/$env/env/
	    git clone https://github.com/johncheetham/breakout.git
	    git clone https://github.com/lukasz1985/PyBlocks.git
	    cd $dirTmp
        fi
        
        # install jupyter notebooks for running install checker notebook
        pip install jupyter
    fi

    # If this is not the ocr environment pass,
    # install the packages in requirements.txt file ...
    if [[ $env == "ocr" ]]; then
    echo ""
    else
    
    # install requirements.txt if this is not the gui pass..
    if [[ $env != "gui" ]]; then
    pip install -r $BASEDIR/install/$env/requirements.txt
    fi
    
    fi

    # check if there is a pythonCode directory for this environment
    # if so copy files from pythonCode to lib/python3.9/site-packages
    directory="$BASEDIR/install/$env/pythonCode"
    if [ -d "$directory" ]; then
        echo "pythonCode directory exists.."
        echo "copying contents to environment's python site-packages directory.."

        # if this is the mmocr env pass..
        # copy in the files in the mmocrPatch directory to the proper location
        # in the mmocr code directory.. (these are Ethan's patch fixes..)
        if [[ $env == "ocr" ]]; then

	# swap the textdet utils.py file for our updated/debugged one (had to debug bool syntax)
	cp $directory/* $BASEDIR/envs/$env/env/lib/python3.8/site-packages/
        mmocrPatchdir="$BASEDIR/install/$env/mmocrPatch"
        cp $mmocrPatchdir/utils.py $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/mmocr/models/textdet/postprocess

	#Add configs dir to the dir above demo so the demo tests work..
	cp -r $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/configs $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/mmocr/
	cp -r $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/configs $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/demo/
	
	# Add the checkpoints directory...	
	mmocrChkpntdir="$BASEDIR/install/$env/mmocrChkpts"
	cp -r $mmocrChkpntdir $BASEDIR/envs/$env/env/

	# Add the updated tutorial jupyter notebook for install testing..
	cp $BASEDIR/install/$env/TextyBeast_MMOCR_Install_Check.ipynb $BASEDIR/envs/$env/env/lib/python3.8/site-packages/mmocr/demo

	else

	# If this is the pyannote pass...
	if [[ $env == "pyannote" ]]; then
	cp $directory/* $BASEDIR/envs/$env/env/lib/python3.8/site-packages/
	fi

	# if this is the whspr pass...
	if [[ $env == "whspr" ]]; then
	cp $directory/* $BASEDIR/envs/$env/env/lib/python3.9/site-packages/
	fi

	# if this is the gui pass...
	if [[ $env == "gui" ]]; then
	cp $directory/* $BASEDIR/envs/$env/env/lib/python3.9/site-packages/
	fi

       fi

    else
        echo "No pythonCode directory present for this environment. Continuing on.. "
    fi

    # deactivate the virtual environment
    deactivate

    # go back to the start dir for the loop ...
    cd $strtDir

done
echo ""
echo "TextyBeast Installation Complete..."
#===============================================================================
