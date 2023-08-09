

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

TextyBeast is a Python and Bash repository for scraping both spoken and written text from video image frames and audio using AI detection, recognition, and transcription models and functions from MMOCR, OpenAI's Whisper, and Pyannote.

This repository can be installed/setup to run jobs locally, on a remote SLURM cluster, or both.

## Requirements/Dependencies:
- Must be running a Linux operating system or MacOS
- Must have an NVIDIA gpu
- Will function best on a Linux machine equipped with newer NVIDIA cards.
- Will need 6-8 GB of GPU ram (preferably more)
- Must have both python3.9 and python3.8 installed
- If installing on a Mac, you may only be able to submit/run jobs remotely on a system equipped with NVIDIA cards. Without dedicated NVIDIA gpus, TextyBeast may run/work but will likely default to much slower cpu-only models.

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

### Running Jobs Locally:

### Running Jobs on Remote Slurm Cluster:

### Running Jobs Remotely on Non-Slurm System::

## Notes

insert notes here...
