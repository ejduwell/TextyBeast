

```
        #########################################################################
        #########################################################################
        ##                                                                     ##
        ##        ::::::::::::::::::::::::    :::::::::::::::::   :::          ##
        ##           :+:    :+:       :+:    :+:    :+:    :+:   :+:           ##
        ##          +:+    +:+        +:+  +:+     +:+     +:+ +:+             ##
        ##         +#+    +#++:++#    +#++:+      +#+      +#++:               ##
        ##        +#+    +#+        +#+  +#+     +#+       +#+                 ##
        ##       #+#    #+#       #+#    #+#    #+#       #+#                  ##
        ##      ###    #############    ###    ###       ###                   ##
        ##            ::::::::: ::::::::::    :::     :::::::::::::::::::      ##
        ##           :+:    :+::+:         :+: :+:  :+:    :+:   :+:           ##
        ##          +:+    +:++:+        +:+   +:+ +:+          +:+            ##
        ##         +#++:++#+ +#++:++#  +#++:++#++:+#++:++#++   +#+             ##
        ##        +#+    +#++#+       +#+     +#+       +#+   +#+              ##
        ##       #+#    #+##+#       #+#     #+##+#    #+#   #+#               ##
        ##      ######### #############     ### ########    ###                ##
        ##                                                                     ##
        ##                                                                     ##
        #########################################################################
        ########################### --- README --- ##############################
        #########################################################################               
```


# Overview:

TextyBeast is a Python and Bash repository for scraping both spoken and written text from video image frames and audio using AI detection, recognition, and transcription models and functions from MMOCR, OpenAI's Whisper, and Pyannote. (see links below)

- https://github.com/open-mmlab/mmocr
- https://github.com/openai/whisper
- https://github.com/pyannote

This repository was developed to scrape text data from video recorded lectures and classroom sessions and provide low-level data describing
spoken and written information within a course/curriculum for educational research and curricular development. This is it's intended use case, 
but it could could be used for other applications as well.

This repository can be installed/setup to run jobs locally, remotely (including SLURM clusters), or both.

## Requirements/Dependencies:
- Must be running a Linux operating system or MacOS
- Must have an NVIDIA gpu
- Will function best on a Linux machine equipped with newer NVIDIA cards.
- Will need 6-8 GB of GPU ram (preferably more)
- Must have both python3.9 and python3.8 installed
- Must have jot installed
- Must have ffmpeg installed
- If installing on a Mac, you may only be able to submit/run jobs remotely on a system equipped with NVIDIA cards. Without dedicated NVIDIA gpus, TextyBeast may run/work but will likely default to much slower cpu-only models.
- Must have a huggingface account and request access to Pyannote's speaker-diarization model:
  
        - Create huggingface account at: https://huggingface.co
        - Get access token for gated pyannote speaker diarization model by following instructions below:
  
                1. visit hf.co/pyannote/speaker-diarization and accept user conditions
                2. visit hf.co/pyannote/segmentation and accept user conditions
                3. visit hf.co/settings/tokens to create an access token
        

- If you're having dependency issues with any of the packages listed above, see the notes section below.
(https://github.com/ejduwell/TextyBeast/blob/main/README.md#notes)

## Installation:
Follow one of the three install instructions below based on which set up you plan to run.

### For running jobs on local machine only:

1. Open a terminal window on your local machine and navigate to desired install location

2. Clone repository:

   ```
   git clone https://github.com/ejduwell/TextyBeast.git
   ```

3. Enter newly created TextyBeast directory:

   ```
   cd TextyBeast
   ```

4. Make installation script executable:

   ```
   chmod +wrx install.sh
   ```

5. Run the install.sh script:

   ```
   ./install.sh
   ```
6. Add local functions to permenant path:
   - Run:
   ```
   cd TextyBeastLocal
   chmod +wrx *
   ./addFcns2Path.sh
   ```
   
8. Test your installation:
    - Return to main TextyBeast directory.
    - Run:
      ```
      ./checkInstall.sh
      ```
    - Follow instructions on command line


### For running jobs on remote SLURM cluster only:

1. Open a terminal window on your local machine and navigate to desired install location

2.  Clone repository:
   
      ```
      git clone https://github.com/ejduwell/TextyBeast.git
      ```

3. Enter TextyBeast/textyBeastLocal directory:
   
   ```
   cd TextyBeast/textyBeastLocal
   ```

4. Make scripts executable:

   ```
   chmod +wrx *
   ```

5. Run setup.sh script:
   
   ```
   ./setup.sh
   ```
   - This procedure sets up info, variables, and ssh keys for submitting remote jobs
   - Follow instructions/provide requested info
   - after completing, run source ~/.bashrc (as suggested in script output)

6. Install TextyBeast on remote system:
   ```
   ./remoteClstrInstall.sh
   ```
7. Add local functions to permenant path:
   - Run:
   ```
   ./addFcns2Path.sh
   ```
### To be able to run jobs locally and on remote SLURM cluster:

-------------------------- Local Setup ---------------------------
1. Open a terminal window on your local machine and navigate to desired install location

2. Clone repository:

   ```
   git clone https://github.com/ejduwell/TextyBeast.git
   ```

4. Enter newly created TextyBeast directory:

   ```
   cd TextyBeast
   ```

6. Make installation script executable:

   ```
   chmod +wrx install.sh
   ```

8. Run the install.sh script:

   ```
   ./install.sh
   ```

10. Test your installation:
   
    - Run:
      ```
      ./checkInstall.sh
      ```
    - Follow instructions on command line

--------------------- Remote Cluster Setup ----------------------

7. Enter TextyBeast/textyBeastLocal directory:
   
   ```
   cd TextyBeast/textyBeastLocal
   ```

8. Make scripts executable:
    ```
    chmod +wrx *
    ```

9. Run setup.sh script:
   ```
   ./setup.sh
   ```
   - This procedure sets up info, variables, and ssh keys for submitting remote jobs
   - Follow instructions/provide requested info
   - after completing, run source ~/.bashrc (as suggested in script output)


10. Install TextyBeast on remote system:
    ```
    ./remoteClstrInstall.sh
    ```
11. Add local functions to permenant path:
   - Run:
   ```
   ./addFcns2Path.sh
   ```    

### To be able to run jobs on a remote machine that is not a SLURM cluster:
   - Same install procedure as for remote SLURM cluster above

## Usage:


### Functions:

There are three seperate command-line functions for running jobs locally, on a remote slurm cluster, and on a remote machine respectively:

**Running Jobs Locally: textyBeast_localjob**

textyBeast_localjob runs jobs on your local machine. 

        textyBeast_localjob <full/path/to/input/dir> <full/path/to/output/dir> <"jobtype"> <"parameterFile">

**Running Jobs on Remote Slurm Cluster: textyBeast_slurmjob**

textyBeast_slurmjob pushes your video data to the slurm cluster specified in the installation procedure via scp, runs the job there, and
then retrieves the outputs to your local machine.

        textyBeast_slurmjob <full/path/to/input/dir> <full/path/to/output/dir> <"jobtype"> <"parameterFile">

**Running Jobs Remotely on Non-Slurm System: textyBeast_remotejob**

textyBeast_slurmjob pushes your video data to the remote machine specified in the installation procedure via scp, runs the job there, and
then retrieves the outputs to your local machine.

        textyBeast_remotejob <full/path/to/input/dir> <full/path/to/output/dir> <"jobtype"> <"parameterFile">

### General Syntax:

The general syntax and options for the three functions are identical. Each require four input arguments:

        Input Arguments:
        
        1) input directory -- full path to a local directory containing the videos you want to process
        2) output directory -- full path to a local directory where you want the TextyBeast to save your results
        3) jobtype -- specifies what type of job you want to run. You can currently choose from one of two options:
        
                - "vl" : indicates you are running a "video lecture" job. 
                
                This mode will first find a minimum set of unique video image frames by running a cross-correlation analysis which essentially 
                looks for time points with large changes. This step is intended to segment the video into unique frames corresponding to the 
                slides. MMOCR functions are used to detect, recognize, and transcribe text present in each unique image frame. Image text is 
                clustered into a human readable order using a clustering routine I developed. Whisper is then used to detect/transcribe text 
                present in the audio extracted from the video file. Finally, the text detected in image frames and audio are combined into a 
                final output .csv containing the start/end time of each unique frame, the text scraped from each image frame, and the text 
                scraped from the audio within each corresponding time window.

                - "di" : indicates you are running a "speaker diarization" job.

                This mode will first extract the audio from the video file and segment the audio into seperate chunks corresponding to 
                seperate speakers detected using Pyannote. The audio chunks are then transcribed using Whisper and assembled into a diarized 
                transcript. Outputs include the diarized transcript text file as well as an html file displaying the the transcript and video. 
                Clicking on text locations on the html page allows users to jump to the corresponding locations within the video.
        
        4) parameterFile -- points to a file containing parameters that control various particular aspects of how MMOCR functions, Whisper, 
                            text clustering, and Pyannote speaker diarization are run. You can input one of two options:

                - "default" : will source the default parameters contained within the file TextyBeast/defaultPars.sh 
                  (contents of defaultPars.sh displayed below)
                
                - "yourCustomPars.sh" : You can also create your own Pars.sh file and adjust the parameter values, models, etc. defined
                inside however you like. To do so, I suggest making a copy of TextyBeast/defaultPars.sh. Give it a unique name, and make sure 
                its saved within the base TextyBeast directory. To use this file instead of the default, simply input the name of the file 
                instead of "default"

        #########################################################################################################
        ##################################### (contents of defaultPars.sh) ######################################
        #########################################################################################################
        
        #!/bin/bash

        # get the full path to the main package directory for this package on this machine
        # Note.. this method assumes you save the defaultPars.sh file in the base TextyBeast directory..
        BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
        homeDir=$BASEDIR;
        
        #-------------------------------------------------------------------------------
        # Specify MMOCR Specific Parameters:
        #-------------------------------------------------------------------------------
        frame_dsrate=5 # Specifies the frame rate at which video is initially sampled
        
        cor_thr=0.95 # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
        
        detector='PANet_IC15' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
        
        recognizer='SAR' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
        
        x_merge=65 # sets the distances at which nearby boxes are merged into one by MMOCR
        
        ClustThr_factor=3 # Sets the distance at which nearby bounding boxes are clustered together (multiple of mean text height)
        
        det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth' #path to local text detector model checkpoint file
        
        recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth' #path to local text recognizer model checkpoint file
        
        #-------------------------------------------------------------------------------
        # Specify Whisper Specific Parameters:
        #-------------------------------------------------------------------------------
        whspModel=base # specify whisper model size (tiny, base, medium, or large) 
        
        #-------------------------------------------------------------------------------
        # Specify Pyannote Specific Parameters:
        #-------------------------------------------------------------------------------
        maxSpeakers=10 # indicate the maximum number of potential seperate speakers in the file.
        
        #-------------------------------------------------------------------------------
        # Export Parameters Specified Above:
        #-------------------------------------------------------------------------------
        export frame_dsrate=$frame_dsrate
        export cor_thr=$cor_thr
        export detector=$detector
        export recognizer=$recognizer
        export x_merge=$x_merge
        export ClustThr_factor=$ClustThr_factor
        export det_ckpt_in=$det_ckpt_in
        export recog_ckpt_in=$recog_ckpt_in
        export whspModel=$whspModel
        export maxSpeakers=$maxSpeakers

        #########################################################################################################
        #########################################################################################################

## Notes:

1) Some local scripts depend on jot. This does not appear to come pre-installed on Ubuntu. If you are running an Ubuntu machine and don't have jot installed yet, you will need to run:
   ```
   sudo apt-get update -y
   sudo apt-get install -y athena-jot
   ```
2) This package depends on ffmpeg. If this is not already installed on your local or remote system, you will need to install it:
   ```
   # If you have sudo privileges:
   #--------------------------------------
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg

   # Centos
   sudo yum install epel-release -y
   sudo rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
   sudo yum install ffmpeg ffmpeg-devel
   #--------------------------------------

   # If you don't have sudo privileges,
   # You can install a local copy from source:
   #--------------------------------------
   # pull local copy to home directory (or wherever you like) and unpack
   cd ~
   wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
   tar xjvf ffmpeg-snapshot.tar.bz2
   cd ffmpeg

   # configure it
   ./configure --prefix=$HOME/ffmpeg_build

   # make/install
   make
   make install

   # add FFmpeg to your PATH so you can run it without specifying the full path. Add this to your ~/.bashrc or ~/.zshrc or appropriate profile file:
   export PATH="$HOME/ffmpeg_build/bin:$PATH"
   source ~/.bashrc
   #--------------------------------------
   ```
3) This package requires both python3.8 and python3.9. If you don't have both of these installed you will need to do so:
   ```
   # If you have sudo privileges:
   #--------------------------------------
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   
   # (For python3.8)
   sudo apt-get install python3.8
   sudo apt-get install python3.8-dev
   
   # (For python3.9)
   sudo apt-get install python3.9
   sudo apt-get install python3.9-dev
   #--------------------------------------

   # If you don't have sudo privileges,
   # You can install a local copy from source:
   #--------------------------------------
   # (for python3.8)
   #download and unpack a local copy of python3.8.17
   wget https://www.python.org/ftp/python/3.8.17/Python-3.8.17.tgz
   tar -xvf Python-3.8.17.tgz

   # enter directory
   cd Python-3.8.17

   # configure it
   ./configure --prefix=${HOME}/python3817

   # run make and then make install to make/install the local copy..
   make
   make install

   # add the python3.8 directory to the permanent general path by running the to the following call
   # which pastes the relevant add-path command within the .bashrc , and then re-source .bashrc
   echo 'export PATH=${HOME}/python3817/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc

   # (for python3.9)
   # follow similar procedure to what is outlined above for python3.8.
   # replace wget call with alternative to pull your desired python3.9 version
   # from https://www.python.org/ftp/python/
   #--------------------------------------
   ```
