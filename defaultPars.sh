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
