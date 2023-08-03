# TextyBeast

Python and Bash repository for scraping both spoken and written text from video and audio using AI models and functions from MMOCR, OpenAI's Whisper, and Pyannote.

This repository can be installed/setup to run jobs locally, on a remote SLURM cluster, or both.

## Requirements/Dependencies:
- Must be running a Linux operating system or MacOS
- Must have an NVIDIA gpu
- Will function best on a Linux machine equipped with newer NVIDIA cards.
- Will need 6-8 GB of GPU ram (preferably more)
- Must have python3.9-dev and python3.8-dev installed
- If installing on a Mac, you may only be able to submit/run jobs remotely on a system equipped with NVIDIA cards. Without dedicated NVIDIA gpus, TextyBeast may run/work but will likely default to much slower cpu-only models.

## Installation:
Follow one of the three install instructions below based on which set up you plan to run.

### For running jobs on local machine only:

1. Open a terminal window on your local machine and navigate to desired install location

2. Run:
   `git clone https://github.com/ejduwell/TextyBeast.git`

3. Run:
   `cd TextyBeast`

4. Run:
   `chmod +wrx install.sh`

5. Run:
   `./install.sh`

6. Test installation
    - Run:
      `./checkInstall.sh`
    - Follow instructions on command line


### For running jobs on remote SLURM cluster only:

1. Open a terminal window on your local machine and navigate to desired install location

2. Run:
   `git clone https://github.com/ejduwell/TextyBeast.git`

4. Run:
   `cd TextyBeast/textyBeastLocal`

6. Run:
   `chmod +wrx *`

8. Run:
   `./setup.sh`
   - Follow instructions/provide requested info
   - after completing, run source ~/.bashrc (as suggested in script output)

6. Run:
   `./remoteClstrInstall.sh`


### To be able to run jobs locally and on remote SLURM cluster:

-------------------------- Local Setup ---------------------------
1. Open a terminal window on your local machine and navigate to desired install location

2. Run:
   `git clone https://github.com/ejduwell/TextyBeast.git`

3. Run:
   `cd TextyBeast`

4. Run:
   `chmod +wrx install.sh`

5. Run:
   `./install.sh`

6. Test installation
    - Run:
      `./checkInstall.sh`
    - Follow instructions on command line

--------------------- Remote Cluster Setup ----------------------

7. cd into TextyBeast/textyBeastLocal (on local machine)

8. Run:
   `chmod +wrx *`

9. Run:
   `./setup.sh`
   - Follow instructions/provide requested info
   - after completing, run source ~/.bashrc (as suggested in script output)

10. Run:
   `./remoteClstrInstall.sh`
   - this script installs TextyBeast on the remote cluster via ssh using the credentials/info provided in previous step (setup.sh)

### Usage:
insert instructions for usage here...

## Notes

insert notes here...
