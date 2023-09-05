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

# LIST OF MMOCR TEXT DETECTOR MODELS AND THEIR CHECKPOINT FILES:
# ----------------------------------------------------------------------
# DB_r18: dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth
# DB_r50: dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth
# DRRG: drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth
# FCE_IC15: fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth
# FCE_CTW_DCNv2: fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth
# MaskRCNN_CTW: mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth
# MaskRCNN_IC15: mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth
# MaskRCNN_IC17: mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth
# PANet_CTW: panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth
# PANet_IC15: panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth
# PS_CTW: psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth
# PS_IC15: psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth
# TextSnake: textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth
# ----------------------------------------------------------------------

# UNCOMMENT PAIRS OF DETECTOR AND DET_CKPT_IN VARIABLES TO SET THE TEXT DETECTION MODEL:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='DB_r18' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
detector='DB_r50' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='DRRG' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='FCE_IC15' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='FCE_CTW_DCNv2' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='MaskRCNN_CTW' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='MaskRCNN_IC15' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='MaskRCNN_IC17' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='PANet_CTW' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='PANet_IC15' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='PS_CTW' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='PS_IC15' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth' #path to local text detector model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#detector='TextSnake' #Text Detection Model Used by MMOCR (finds and places bounding boxes around text)
#det_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth' #path to local text detector model checkpoint file


# LIST OF MMOCR TEXT RECOGNITION MODELS AND THEIR CHECKPOINT FILES:
# ----------------------------------------------------------------------
# ABINet: abinet_academic-f718abf6.pth
# CRNN: crnn_academic-a723a1c5.pth
# SAR: sar_r31_parallel_decoder_academic-dba3a4a3.pth
# SAR_CN*
# NRTR_1/16-1/8: nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth
# NRTR_1/8-1/4: nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth
# RobustScanner: robustscanner_r31_academic-5f05874f.pth
# SATRN: satrn_academic_20211009-cb8b1580.pth
# SATRN_sm: satrn_small_20211009-2cf13355.pth
# SEG: seg_r31_1by16_fpnocr_academic-72235b11.pth
# CRNN_TPS: crnn_tps_academic_dataset_20210510-d221a905.pth
# ----------------------------------------------------------------------

# UNCOMMENT PAIRS OF RECOGNIZER AND RECOG_CKPT_IN VARIABLES TO SET THE TEXT RECOGNITION MODEL:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='ABINet' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/abinet_academic-f718abf6.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='CRNN' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/crnn_academic-a723a1c5.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
recognizer='SAR' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='NRTR_1/16-1/8' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='NRTR_1/8-1/4' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='RobustScanner' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/robustscanner_r31_academic-5f05874f.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='SATRN' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/satrn_academic_20211009-cb8b1580.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='SATRN_sm' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/satrn_small_20211009-2cf13355.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='SEG' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/seg_r31_1by16_fpnocr_academic-72235b11.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#recognizer='CRNN_TPS' #Text Recognition Model Used by MMOCR (recognizes/interprets text within detected regions)
#recog_ckpt_in=$homeDir'/envs/ocr/env/mmocrChkpts/crnn_tps_academic_dataset_20210510-d221a905.pth' #path to local text recognizer model checkpoint file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

x_merge=65 # sets the distances at which nearby boxes are merged into one by MMOCR

ClustThr_factor=3 # Sets the distance at which nearby bounding boxes are clustered together (multiple of mean text height)

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
