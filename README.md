# TextyBeast

Python repository for scraping both spoken and written text from video and audio using AI models and functions from MMOCR, OpenAI's Whisper, and Pyannote.
This repository can be installed/setup to run jobs locally, on a remote SLURM cluster, or both.

### Requirements/Dependencies
- must be running linux operating system or MacOs
- must have python3.9-dev and python3.8-dev installed

### Installation

(LOCAL ONLY)
------------------------------------------------------------------
1. Open terminal window and navigate to desired install location

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

(REMOTE CLUSTER ONLY)
------------------------------------------------------------------
1. Open terminal window and navigate to desired install location

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
------------------------------------------------------------------

(LOCAL & REMOTE CLUSTER)
------------------------------------------------------------------
# ------ Local Setup ------ 
1. Open terminal window and navigate to desired install location

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

# ------ Remote Cluster Setup ------
7. cd into TextyBeast/textyBeastLocal

8. Run:
   `chmod +wrx *`

9. Run:
   `./setup.sh`
   - Follow instructions/provide requested info
   - after completing, run source ~/.bashrc (as suggested in script output)

10. Run:
   `./remoteClstrInstall.sh`
------------------------------------------------------------------

### Usage
insert instructions for usage here...

## Notes

insert notes here...
