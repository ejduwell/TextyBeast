#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:30:19 2022

@author: eduwell
"""

#%% Import Packages
import os
import pdf2txt
import lecxr_text
import txt_cleaning_spchk
def vid_txtskrp_val(pdf_file,vid_file, pdf_dir, vid_dir,pdftxtout,vidtxtout,outdir):
    
    #%% Parameters
    
    scrape_imgs = 0;
    x_merge = 65; # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box.
    ClustThr_factor = 3; # controls agglomerative clustering threshold distance (in units of n*average textbox height)
    cor_thr = 0.85; # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
    frame_dsrate = 10; # Specifies the frame rate at which video is initially sampled
    detector='PANet_IC15'; # specifies detector to be used used by mmocr to detect/put bounding boxes around text
    #detector='TextSnake'
    recognizer='SAR'; # specifies the recognizer which interprets/"reads" the text image within bounding boxes
    
    #%% extract text from veridical pdf slide file
    pdf2txt.pdf2txt(pdf_file, pdf_dir, pdftxtout, outdir, 0)

    #%% extract text from corresponding video lecture frames
    lecxr_text.extract_all(vid_dir, vid_file, frame_dsrate, cor_thr, detector, recognizer, x_merge, ClustThr_factor)
    
    
    #%% Read in the .txt files from veridical pdf and video 
   
    suggest_corrected = 11 # 0: dont suggest corrections, 
                            # 1: suggest the top suggested correction, 
                            # 2: provide list of suggestion candidates.
                            # 3: provide only the lists of known words and unknown words
                            # 4: check unknown words against a selected pyset_dictionary .txt file using dictionary_chkr.
                            #    add "unknown" words which match entries in the dictionary into the "known words" set
                            # 5: check unknown words against a selected pyset_dictionary .txt file using 
                            #    combined with spellchecker english dictionary. Suggest the top suggested correction
                            # 6: check unknown words against a selected pyset_dictionary .txt file using 
                            #    combined with spellchecker english dictionary. Provides list of suggestion candidates.
                            # 7: use existing .json dictionary (ie such as the as the guttenberg one from: https://github.com/dwyl/english-words)
                            #    Suggest the top suggested correction
                            # 8: use existing .json dictionary (ie such as the as the guttenberg one from: https://github.com/dwyl/english-words)
                            #    Provide list of suggestion candidates.
                            # 9: same as 7 but .json is loaded on top of base spellchecker() dictionary..
                            # 10: same as 8 but .json is loaded on top of base spellchecker() dictionary..
                            # 11: same as 9 but also adds an additional custom .txt file in too
                            # 12: same as 10 but also adds an additional custom .txt file in too
    
    dict_file = ["words_dictionary.json", "pyset_dictnry_webster_netter_medterm.txt"]
    dict_filedir = ["/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/english-words-master/ethans_selections", "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/tst_2"]
    words_out_pdf, words_out_unique_pdf, words_out_unknown_pdf, words_out_unknownWcorr_pdf, words_out_known_pdf, dicnry = txt_cleaning_spchk.getwords(outdir,pdftxtout,suggest_corrected,dict_file,dict_filedir,0)
    
    words_out_vid, words_out_unique_vid, words_out_unknown_vid, words_out_unknownWcorr_vid, words_out_known_vid, dicnry = txt_cleaning_spchk.getwords(outdir,vidtxtout,suggest_corrected,dict_file,dict_filedir,0)
    
    
    
#%% Test Call
# pdf params
pdf_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/"
out_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/"
pdf_file = "";
pdftxtout = "test_pdf_txt.txt";

# video params
vid_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/Course_Materials_Data/CHA/CHA1-2_VIDEO/test_trimmed/";
# NOTE : ALWAYS END PATH STRINGS FOR DIRECTORIES WITH A "/".. had to chose a convention for concatenating stuff...
vid_file = "CHA_I_Introduction_to_the_Nervous_System_default.mp4";
vidtxtout = "test_vid_txt.txt";

# other params
outdir =""
    
vid_txtskrp_val(pdf_file,vid_file, pdf_dir, vid_dir,pdftxtout,vidtxtout,outdir)
    
    
    
    