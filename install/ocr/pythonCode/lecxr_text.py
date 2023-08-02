#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 20:03:12 2022


READ UP ON:
    https://audiosegment.readthedocs.io/en/latest/audiosegment.html
    
@author: eduwell
"""
#%% Import Packages
##
import img2txt
import speech2text_al
import frm2txt_DL
import frm2txt_DL2
import pdf2txt
import im_textclean as imtc

import os
#import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
import shutil
from pydub import AudioSegment
#from pydub.silence import split_on_silence
import wave
import contextlib
#from scipy.io.wavfile import read
from scipy import stats
import peakutils


import pdfplumber
#from gingerit.gingerit import GingerIt
import copy
#%% Parameters
config_dir = "/home/eduwell/python_projects/data_scraping/mmocr/configs"
db_skip = 1; # for debugging purposes.. place if statement prior to code you want to skip checking for this to skip stuff..
#%% Functions
def jpg2txt_tsrc(jpg_file,outdir):
    # Import libraries
    from PIL import Image
    import pytesseract
    import sys
    from pdf2image import convert_from_path
    import os
     
    # Path of the pdf
    #PDF_file = "d.pdf"
      
    '''
    Part #1 : Converting PDF to images
    '''
    
    #EJD: COMMENTED.. feeding in an image.. don't need to convert..
    
    os.chdir(outdir) 
    
      
    '''
    Part #2 - Recognizing text from the images using OCR
    '''

    # Variable to get count of total number of pages
    #filelimit = image_counter-1
      
    # Creating a text file to write the output
    outfile = os.path.splitext(jpg_file)[0]+".txt"
    #outfile = "out_text.txt"
      
    # Open the file in append mode so that 
    # All contents of all images are added to the same file
    f = open(outfile, "a")
      
    # Iterate from 1 to total number of pages
    #for i in range(1, filelimit + 1):
      
    # Set filename to recognize text from
    # Again, these files will be:
    # page_1.jpg
    # page_2.jpg
    # ....
    # page_n.jpg
    
    filename = jpg_file
          
    # Recognize the text as string in image using pytesserct
    text = str(((pytesseract.image_to_string(Image.open(filename)))))
  
    # The recognized text is stored in variable text
    # Any string processing may be applied on text
    # Here, basic formatting has been done:
    # In many PDFs, at line ending, if a word can't
    # be written fully, a 'hyphen' is added.
    # The rest of the word is written in the next line
    # Eg: This is a sample text this word here GeeksF-
    # orGeeks is half on first line, remaining on next.
    # To remove this, we replace every '-\n' to ''.
    text = text.replace('-\n', '')
    
    #EJD ADDED BELOW TO TRY TO KEEP ALL TEXT FROM EACH CLUSTER SAVED ON A SINGLE LINE (REMOVE THE NEXT LINE PROMPTS..)
    #text = text.replace('\n', ' ')
    
    # Finally, write the processed text to the file.
    f.write(text)
      
    # Close the file after writing all the text.
    f.close()
def extract_all(vid_dir, video, frame_dsrate, cor_thr, detector, recognizer, x_merge, ClustThr_factor):
    
    audio_db = 0;
    
    # Go to video directory
    os.chdir(vid_dir)
    
    print("*****SETTING UP DIRECTORIES*****")
    out_dir = "out_dir"
    # Set up output directory structure and copy video file in
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    source = video
    destination = out_dir + '/'+ video
    shutil.copy(source, destination)
    
    os.chdir(out_dir)
    root_dir = os.getcwd()
    
    vid_imdir = root_dir+"/video_img_dta/"
    if not os.path.exists(vid_imdir):
        os.makedirs(vid_imdir)
    
    vid_spdir = root_dir+"/audio_speech_dta/"
    if not os.path.exists(vid_spdir):
        os.makedirs(vid_spdir)
     
    print("*****EXTRACTING AUDIO FROM VIDEO FILE*****") 
    # Extract .WAV audio from video and save in audio directory
    clip = mp.VideoFileClip(video) 
    audioOut_fname = os.path.splitext(video)[0]+".wav" 
    
    # enter audio directory, convert video to audio, and save as .wav file
    os.chdir(vid_spdir)
    #clip.audio.write_audiofile(audioOut_fname)
    clip.audio.to_audiofile(audioOut_fname)
    os.chdir(root_dir)
    
    print("*****EXTRACTING IMAGE FRAMES FROM VIDEO*****")

    # Convert video into image frames
    ds = 1
    img2txt.vid2frms(root_dir,video,vid_imdir,frame_dsrate,ds)
    print(" ")
    
    if audio_db ==0:
        print("*****FINDING MINIMUM SET OF UNIQUE FRAMES*****")
        begin_time = time.perf_counter()
        # Reduce image frame set to minimum number of unique frames
        # Save unique frames in new directory
        uf_outdir = "unique_frames"
        tag = "*frame_*";
        img2txt.unique_frms(vid_imdir,uf_outdir, tag, cor_thr)
        end_time = time.perf_counter()
        time_elapsed = end_time - begin_time
        time_elapsed = str(time_elapsed)
        end_time_message = "TOOK"+" "+time_elapsed+" "+"SECONDS"
        print(end_time_message)
        print(" ")
        
        # Copy in the "configs" directory for mmocr
        source_config = config_dir
        
        destination_config = vid_imdir+"unique_frames/configs"
        if not os.path.exists(destination_config):
            shutil.copytree(source_config, destination_config)
        
        
        print("*****EXTRACTING TEXT FROM IMAGE FRAMES WITH MMOCR*****")
        begin_time = time.perf_counter()
        # run individual frames through mmocr to extract text
        os.chdir(vid_imdir+uf_outdir)
        mmocr_dir = "mmocr_out/"
        if not os.path.exists(mmocr_dir):
            os.makedirs(mmocr_dir)
        
            
        
        
        # EJD COMMENTED TO TRY NEW MIXED METHOD OF TEXT SNAKE DETECT/TESSERACT RECOGNIZE I DEVELOPED IN PDF2TXT... 
        #frm2txt_DL.frm2txt_mmocr(vid_imdir+uf_outdir,tag,mmocr_dir, detector, recognizer)
        
        

        # Run text cleaner algorithm from fred's image magic recursively over each file to prepare images for ocr..
        #imtc.im_textclean_rc(vid_imdir+uf_outdir,vid_imdir+uf_outdir,"*_sec_mean.jpeg*")
        
        
        
        #Run each cleaned frame through the mmocr text detection algoritm to detect and bound text with "textsnake"
        
        #ORIG
        #frm2txt_DL.frm2txt_mmocr_det(vid_imdir+uf_outdir,'*_sec_mean_tc.jpeg*',mmocr_dir, detector)
        
        
        
        
        #!!!!!
        #NEW TEST# #MAKE SURE TO UNCOMMENT!!!!!
        #frm2txt_DL2.frm2txt_mmocr_det(vid_imdir+uf_outdir,'*_sec_mean_tc.jpeg*',mmocr_dir, detector,recognizer)
        
        #trying w/out text cleaning pre-processing step
        frm2txt_DL2.frm2txt_mmocr_det(vid_imdir+uf_outdir,'*_sec_mean.jpeg*',mmocr_dir, detector,recognizer,x_merge)
        #!!!!!
        
        
        
        #ClustThr_factor = 1; # Specify Cluster Thresholding Factor (max distance clusters of text can be apart and still cluster together.. expressed as a multiple of the average text bounding box height within each image)
        
        # For each frame iterate through all textsnake bounding boxes, cluster them spatially, and them mask each cluster individually against a white background
        
        #ORIG#
        #frm2txt_DL.bbox_txtMask_snek(vid_imdir+uf_outdir+"/"+mmocr_dir, vid_imdir+uf_outdir+"/"+mmocr_dir, "*_mmorc_TextSnake.jpg*","*_sec_mean_tc.json*", ClustThr_factor, "euclidean", "ward")
        
        
        #NEW TEST#
        #txt_out = frm2txt_DL2.bbox_txtMask_snek(vid_imdir+uf_outdir+"/"+mmocr_dir, vid_imdir+uf_outdir+"/"+mmocr_dir, "*_mmorc_PANet_IC15.png*","*_sec_mean.json*", ClustThr_factor, "euclidean", "ward")
        txt_out = frm2txt_DL2.bbox_txtMask_snek(vid_imdir+uf_outdir+"/"+mmocr_dir, vid_imdir+uf_outdir+"/"+mmocr_dir, "*.png*","*_sec_mean.json*", ClustThr_factor, "euclidean", "ward")
        #txt_out = frm2txt_DL2.bbox_txtMask_snek(vid_imdir+uf_outdir+"/"+mmocr_dir, vid_imdir+uf_outdir+"/"+mmocr_dir, "*_mmorc_PANet_IC15.jpg*","*_sec_mean_tc.json*", ClustThr_factor, "euclidean", "ward")
        
        
        
        

        
    
        
        txt_outname = os.path.splitext(video)[0]+"_framesText.txt"
        txt_outname_fxr = os.path.splitext(video)[0]+"_framesText_ocrFixr.txt"
        
        #GINGERIT STUFF:
        # txt_out_sp = copy.deepcopy(txt_out)
        # parser = GingerIt()

        # frame_itr = 0;
        # for frames in txt_out_sp:
        #     clst_itr = 0
        #     for txt_clusters in frames:
        #         subclst_itr = 0
        #         for sub_clustrs in txt_clusters:
        #             str_itr = 0
        #             clst_str =""
        #             for txt_strgs in sub_clustrs:
        #                 txt2read = txt_strgs
        #                 if str_itr > 0:
        #                     clst_str = clst_str+" "+txt2read
        #                 else:
        #                     clst_str = clst_str+txt2read     
        #                 str_itr = str_itr+1
        #             txt2read_Ginger = parser.parse(clst_str)
        #             txt2read_Ginger_result = txt2read_Ginger['result']
        #             txt_out_sp[frame_itr][clst_itr][subclst_itr]= [txt2read_Ginger_result]
                        
        #             subclst_itr = subclst_itr+1
        #         clst_itr = clst_itr+1
        #     frame_itr = frame_itr+1
        
        
        # Attempt to spellcheck the text_outname with "gingerit"
        # txt_out_sp = copy.deepcopy(txt_out)
        # parser = GingerIt()

        # frame_itr = 0;
        # for frames in txt_out_sp:
        #     clst_itr = 0
        #     for txt_clusters in frames:
        #         subclst_itr = 0
        #         for sub_clustrs in txt_clusters:
        #             str_itr = 0
        #             clst_str =""
        #             for txt_strgs in sub_clustrs:
        #                 txt2read = txt_strgs
        #                 if str_itr > 0:
        #                     clst_str = clst_str+" "+txt2read
        #                 else:
        #                     clst_str = clst_str+txt2read     
        #                 str_itr = str_itr+1
        #             txt2read_Ginger = parser.parse(clst_str)
        #             txt2read_Ginger_result = txt2read_Ginger['result']
        #             txt_out_sp[frame_itr][clst_itr][subclst_itr]= [txt2read_Ginger_result]
                        
        #             subclst_itr = subclst_itr+1
        #         clst_itr = clst_itr+1
        #     frame_itr = frame_itr+1
        
        
        frame_itr = 0;
        with open(txt_outname, 'w') as f:
            for frames in txt_out:
                frame_str = "TEXT SCRAPED FROM UNIQUE FRAME #"+" "+str(frame_itr).zfill(4)
                strz = "****************************************************************************************"
                
                print(strz)
                print('\n')
                print(strz)
                print('\n')
                print(frame_str)
                print('\n')
                print(strz)
                print('\n')
                print(strz)
                print('\n')
                
                
                f.write(strz)
                f.write('\n')
                f.write(strz)
                f.write('\n')
                f.write(frame_str)
                f.write('\n')
                f.write(strz)
                f.write('\n')
                f.write(strz)
                f.write('\n')
                f.write('\n')
                
                #EJD COMMENTED
                #frame_base = os.path.splitext(frames)[0]
                
                # #del ii
                # with pdfplumber.open(frame_base+".pdf") as pdf:
                #     pages = pdf.pages
                #     n_pages = len(pages)
                #     f.write("-------------------------------")
                #     f.write('\n')
                #     f.write("PLAIN_TEXT_READ_FROM_THIS_PAGE:")
                #     f.write('\n')
                #     f.write("-------------------------------")
                #     f.write('\n')
                #     f.write('\n')
                #     page = pdf.pages[0]
                #     f.write(page.extract_text())
                #     f.write('\n')
                #     f.write('\n')
                #     print(page.extract_text())
                #     print(" ")
                
                    
                
               #EJD COMMENTED
                # frame_tag = "*"+frame_base+"_mmorc_TextSnake_"+"*"
                
                # #Get list of text cluster masked images containing frame tag..
                # frame_tag_lst = glob.glob(frame_tag)
                # frame_tag_lst.sort()
                
                # frame_itr = frame_itr+1
                
                # Loop through txt_out and print contained text into output .txt file
                cntr = 0
                for txt_clusters in frames:
                    for sub_clustrs in txt_clusters:
                        if cntr > 0:
                            f.write('\n')
                        for txt_strgs in sub_clustrs:
                            txt2read = txt_strgs
                            f.write(txt2read)
                            f.write('\n')
                    f.write('\n')
                    f.write('\n')
                    cntr = cntr+1
                frame_itr = frame_itr+1
        f.close()
        pause = ""
        
        # Attempt to spellcheck the text_outname with "OCRfixr"
        #ocrfxr_cmd = "ocrfixr "+txt_outname+" "+txt_outname_fxr
        #os.system(ocrfxr_cmd)
        
        

        # ORIGINAL TXT READOUT WITH SEP IMS AND TSRACT.. :
        ###############################################################################   
        # # Get Number of Unique Frames
        # os.chdir(vid_imdir+uf_outdir)
        
        # uf_list_base = glob.glob("*_tc.jpeg*")
        
        # uf_list_base.sort() # make sure they are sorted in ascending order
        
        # # Go back to mmocr output directory and loop through unique frames to extract
        # # text from individual text cluster masked images ...
        
        # os.chdir(vid_imdir+uf_outdir+"/"+mmocr_dir)
        
        # txt_outname = os.path.splitext(video)[0]+"_framesText.txt"
        # frame_itr = 0;
        # with open(txt_outname, 'w') as f:
        #     for frames in uf_list_base:
        #         frame_str = "TEXT SCRAPED FROM UNIQUE FRAME #"+" "+str(frame_itr).zfill(4)+" "+"("+frames+"):"
        #         strz = "****************************************************************************************"
                
        #         print(strz)
        #         print('\n')
        #         print(strz)
        #         print('\n')
        #         print(frame_str)
        #         print('\n')
        #         print(strz)
        #         print('\n')
        #         print(strz)
        #         print('\n')
                
                
        #         f.write(strz)
        #         f.write('\n')
        #         f.write(strz)
        #         f.write('\n')
        #         f.write(frame_str)
        #         f.write('\n')
        #         f.write(strz)
        #         f.write('\n')
        #         f.write(strz)
        #         f.write('\n')
        #         f.write('\n')
                
        #         frame_base = os.path.splitext(frames)[0]
        #         # #del ii
        #         # with pdfplumber.open(frame_base+".pdf") as pdf:
        #         #     pages = pdf.pages
        #         #     n_pages = len(pages)
        #         #     f.write("-------------------------------")
        #         #     f.write('\n')
        #         #     f.write("PLAIN_TEXT_READ_FROM_THIS_PAGE:")
        #         #     f.write('\n')
        #         #     f.write("-------------------------------")
        #         #     f.write('\n')
        #         #     f.write('\n')
        #         #     page = pdf.pages[0]
        #         #     f.write(page.extract_text())
        #         #     f.write('\n')
        #         #     f.write('\n')
        #         #     print(page.extract_text())
        #         #     print(" ")
                
                    
                
               
        #         frame_tag = "*"+frame_base+"_mmorc_TextSnake_"+"*"
                
        #         #Get list of text cluster masked images containing frame tag..
        #         frame_tag_lst = glob.glob(frame_tag)
        #         frame_tag_lst.sort()
                
        #         frame_itr = frame_itr+1
                
        #         # Loop through images in frame_tag_lst and extract text with tesseract...
        #         for txt_clusters in frame_tag_lst:
        #             jpg2txt_tsrc(txt_clusters,vid_imdir+uf_outdir+"/"+mmocr_dir) 
        #             tfile = os.path.splitext(txt_clusters)[0]+".txt"
                    
        #             with open(tfile) as tf:
        #                   imtxtlines = tf.readlines()
                    
        #             if len(imtxtlines) > 0:
        #                 f.write('\n')  
        #                 for bb in imtxtlines:
        #                       f.write(bb)
        #                       f.write('\n')
        #             else:
        #                 f.write('\n')
                    
        #             rm_str = "rm"+" "+tfile
        #             rm_str2 = "rm"+" "+txt_clusters
        #             os.system(rm_str)
        #             os.system(rm_str2)
###############################################################################                
                    
        # # Get list of text-masked images
        # mfile_list = glob.glob("*_mask_apld.jpg*")

        # mfile_list.sort() # make sure they are sorted in ascending order
        
        

        #     for files in mfile_list:
                
        #         pdf2txt.jpg2txt_tsrc(files,vid_imdir+uf_outdir)                      
                
        #         tfile = os.path.splitext(files)[0]+".txt"
                
        #         with open(tfile) as tf:
        #              imtxtlines = tf.readlines()
                
        #         if len(imtxtlines) > 0:
        #             f.write('\n')  
        #             for bb in imtxtlines:
        #                  f.write(bb) 
                
        #         rm_str = "rm"+" "+tfile
        #         rm_str2 = "rm"+" "+files
        #         os.system(rm_str)
        #         os.system(rm_str2)
        
        os.chdir(root_dir)
        end_time = time.perf_counter()
        time_elapsed = end_time - begin_time
        time_elapsed = str(time_elapsed)
        end_time_message = "TOOK"+" "+time_elapsed+" "+"SECONDS"
        print(end_time_message)
        print(" ")
        
        # use bounding boxes from mmocr to mask out "text only" images
        # STILL NEED TO WRITE THIS CODE...
        #im_dir = vid_imdir+"/"+"unique_frames/"
        #txt_dir = vid_imdir+"/"+"unique_frames/mmocr_out/"    
        #im_tag = "*.jpeg*"
        #txt_tag = "*.json*"
        #frm2txt_DL.bbox_txtMask(im_dir, txt_dir, im_tag,txt_tag)
        
        # run the "text only" images through tesseract pipline to extract text
        # a second time for cross examination of mmocr deep learning extraction
        # STILL NEED TO WRITE THIS CODE...
        #tag = "*_mask_apld.jpg*"
        #img2txt.ts_all(im_dir,tag, txt_dir, 0, 0)
   
    if db_skip == 0: # placed for debugging purposes...
        # NOTE WILL NEED TO UNINDENT ALL BELOW WHEN DONE DEBUGGING (IF YOU REMOVE THE IF ABOVE..)
        print("DIVIDING THE AUDIO FILE INTO BITE-SIZED CHUNKS FOR SPEECH RECOGNITION")
        os.chdir(vid_spdir)
        
    
        if not os.path.exists("audio_chunks"):
            os.makedirs("audio_chunks")
        
        aud_chunks_dir = vid_spdir+"audio_chunks/"
        
        def get_audio_len(fname):
            with contextlib.closing(wave.open(fname,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            return duration
        
        # Define a function to normalize a chunk to a target amplitude.
        def match_target_amplitude(aChunk, target_dBFS):
            ''' Normalize given audio chunk '''
            change_in_dBFS = target_dBFS - aChunk.dBFS
            return aChunk.apply_gain(change_in_dBFS)
        
        def audio_seg_dbfs(audio_clip, t1, t2):
            audio_segment = audio_clip[t1:t2]
            seg_dbfs = audio_segment.dBFS
            return seg_dbfs
        
        def sliding_win_dbfs(audio_clip, win_len, step_size):
            #audio_clip = AudioSegment.from_wav(audio_clip)
            clip_len = len(audio_clip)
            n_points = round((clip_len-win_len)/step_size)
            db_tseries = np.empty(shape=(1,n_points),dtype='float32')
            x_axis = np.empty(shape=(1,n_points),dtype='float32')
            oo = 0
            for ii in range(0,(n_points-1)):
                t1 = oo
                t2 = oo+win_len
                #audio_segment = audio_clip[t1:t2]
                seg_dbfs = audio_seg_dbfs(audio_clip,t1, t2)
                db_tseries[0][ii] = seg_dbfs
                x_axis[0][ii] = ii
                oo = oo+step_size
            return db_tseries, x_axis
        
        def silence_counter(audio_clip,win_len,s_thr):
            counter = 0
            binary_switch = 0
            
            step_size = 1
            dbfs_tseries, x_axis = sliding_win_dbfs(audio_clip, win_len, step_size)
            clip_len = len(audio_clip)
            n_points = round((clip_len-win_len)/step_size)
            
            for ii in range(0,dbfs_tseries.shape[0]):
                if (abs(dbfs_tseries[0][ii]) <= s_thr):
                    xc = ii
                    plt.axvline(x=xc, color='r', linestyle='--')
                    if binary_switch != 1:
                        counter = counter+1
                        binary_switch = 1
                else:
                    binary_switch = 0
            return counter
        
        def silence_counter2(time_series,baseline_vector):
            index_vect = np.empty(shape=(1,time_series.shape[0]))
            dbfs_tseries = time_series
            step_size = 1
            clip_len = len(audio_clip)
            n_points = round((clip_len-win_len)/step_size)
            
            for ii in range(0,dbfs_tseries.shape[0]-1):
                if dbfs_tseries[ii] <= (baseline_vector[ii]+baseline_vector.std()):
                    index_vect[0][ii] = 70
                else:
                    index_vect[0][ii] = 0
    
            return index_vect
        
        def cut_points(index_vect, min_len):
            nz_indices = np.nonzero(index_vect)
            cut_pts = np.empty(shape=(1,nz_indices[0].shape[0]))
            cut_pts[0][:] = nz_indices[0][:]
            for ii in range(0,nz_indices[0].shape[0]-1):
                point = nz_indices[0][ii]
                next_point = nz_indices[0][ii+1]
                if (next_point == point+1) or (next_point < point+min_len):
                    cut_pts[0][ii+1] = 0
                    
            cut_pts = cut_pts[0][:]
            cut_pts = cut_pts[cut_pts > 0]
            
            return cut_pts
        
        def silence_gridSearch(audio_clip, s_length_min, s_length_max, s_length_grain, s_thr_min, s_thr_max, s_thr_grain):
            s_lengths = np.arange(s_length_min,s_length_max,s_length_grain)
            s_thrlds = np.arange(s_thr_min,s_thr_max,s_thr_grain)
            silence_len_and_thr_choice_grid = np.empty(shape=(s_lengths.shape[0],s_thrlds.shape[0]),dtype='uint8')
            
            for ii in range(0,(s_lengths.shape[0]-1)):
                begin_time = time.perf_counter()
                for oo in range(0,(s_thrlds.shape[0]-1)):
                    silence_len_and_thr_choice_grid[ii][oo] = silence_counter(audio_clip,s_lengths[ii],s_thrlds[oo])
                
                end_time = time.perf_counter()
                time_elapsed = end_time - begin_time
                time_elapsed = str(time_elapsed)
                ii_max_str = str(s_lengths.shape[0]-1)
                ii_str = str(ii)
                end_time_message = ii_str+" "+"OF"+" "+ii_max_str+" "+"TOOK"+" "+time_elapsed+" "+"SECONDS"
                print(end_time_message)
                junk = "jnk"
            return silence_len_and_thr_choice_grid
          
        # Load your audio.
        audio_clip = AudioSegment.from_wav(audioOut_fname)
        
        #win_len = 1000
        win_len = 750
        step_size = 1
        
        
        db_tseries, x_axis = sliding_win_dbfs(audio_clip, win_len, step_size)
        # Creating numpy array
        
        lastElementIndex_y_min1 = (db_tseries.shape[1]-1)
        #lastElementIndex_x = len(x_axis)-1
        
        # Removing the last element using slicing 
        db_tseries_delLast = db_tseries[0][:lastElementIndex_y_min1]
        #x_axis_delLast  = db_tseries_test[:lastElementIndex_x]
        db_tseries_delLast_abs = abs(db_tseries_delLast)
        db_tseries_delLast_abs_round = np.around(db_tseries_delLast_abs, decimals=0)
        
        #db_tseries_delLast_abs_noinf = db_tseries_delLast_abs[db_tseries_delLast_abs < 1E308] # use whatever threshold you like
        
        db_tseries_delLast_abs_noinf = np.array(db_tseries_delLast_abs)
        db_tseries_delLast_abs_noinf0s = np.array(db_tseries_delLast_abs)
        db_tseries_delLast_abs_noinf0s = db_tseries_delLast_abs_noinf0s[db_tseries_delLast_abs < 1E308] # use whatever threshold you like
        
        mean_value = db_tseries_delLast_abs_noinf0s.mean()
        
        #db_tseries_delLast_abs_noinf[db_tseries_delLast_abs == 'NaN'] = round(mean_value)
        db_tseries_delLast_abs_noinf[db_tseries_delLast_abs >= 1E308] = round(mean_value)
    
        xvals = list(range(0,len(db_tseries_delLast_abs_noinf)))
        baseline_values = peakutils.baseline(db_tseries_delLast_abs_noinf)
        #baseline_incl_lim = baseline_values+baseline_values.std()
        bl_check = baseline_values - db_tseries_delLast_abs_noinf
        bl_check[bl_check >= 0] = 70
        bl_check[bl_check <= 0] = 0
    
        #cut_pts = silence_counter2(db_tseries_delLast_abs_noinf,baseline_values)
        
        
        mode_value = stats.mode(db_tseries_delLast_abs_noinf)
        median_value = np.median(db_tseries_delLast_abs_noinf)
        mean_value = np.mean(db_tseries_delLast_abs_noinf)
        
        s_pad = 500
        s_pad_txt = str(s_pad)
        silence_chunk = AudioSegment.silent(duration=s_pad)
        target_amp_norm = -20
        target_amp_norm_txt = str(abs(target_amp_norm))
        chunk_string_base = os.path.splitext(audioOut_fname)[0]
        
        c_pts = cut_points(bl_check, 5000)
        #!!!!!! may want to add 1/2 window length to theses...
        os.chdir(aud_chunks_dir)
        for ii in range(0, c_pts.shape[0]):
            if ii == 0:
                t1 = 0
                t2 = c_pts[ii]
            else:
                t1 = c_pts[ii-1]
                t2 = c_pts[ii]
            
            t1_text = str(round(t1))
            t2_text = str(round(t2))
            
            chunk = audio_clip[t1:t2]
            
            # Add the padding chunk to beginning and end of the entire chunk.
            chunk = silence_chunk + chunk + silence_chunk
            
            # Normalize the entire chunk.
            chunk = match_target_amplitude(chunk, target_amp_norm)
            
            print("Exporting chunk{0}.wav.".format(ii))
            chunk.export(
                ".//"+str(ii).zfill(9)+"_"+chunk_string_base+"_chunk_spad_"+s_pad_txt+"_norm_"+target_amp_norm_txt+"_"+t1_text+"_to_"+t2_text+".wav",
                bitrate = "192k",
                format = "wav"
            )
        
        # get the last chunk at the end too...
        ii = ii+1
        t1 = c_pts[-1]
        t2 = len(audio_clip)
        t1_text = str(round(t1))
        t2_text = str(round(t2))
        chunk = audio_clip[t1:t2]
        # Add the padding chunk to beginning and end of the entire chunk.
        chunk = silence_chunk + chunk + silence_chunk
        # Normalize the entire chunk.
        chunk = match_target_amplitude(chunk, target_amp_norm)
        chunk.export(
            ".//"+str(ii).zfill(9)+"_"+chunk_string_base+"_chunk_spad_"+s_pad_txt+"_norm_"+target_amp_norm_txt+"_"+t1_text+"_to_"+t2_text+".wav",
            bitrate = "192k",
            format = "wav"
        )
        
        fig1 = plt.figure(figsize=(120, 16))
        plt.plot(xvals,db_tseries_delLast_abs_noinf, label='db_tseries_test', linewidth=3)
        plt.plot(xvals,baseline_values, 'r', label='baseline', linewidth=3)
        plt.plot(bl_check,'g', label='cut_points', linewidth=3)
        
        for xc in c_pts:
            plt.axvline(x=xc, color='k', linestyle='--',linewidth=3)
        
        plt.legend()
        plt.show()
     
        
        # bins = np.arange(0,100,0.5)
        # fig2 = plt.figure(figsize=(120, 16))
        # plt.hist(db_tseries_delLast_abs_noinf, bins)
        # plt.show()
        
        
    
        
    
        #%% OLD METHOD OF SILENCE SPLITTING.. (EJD COMMENTED OUT BC HE WROTE HIS OWN ABOVE)
        # Split track where the silence is 2 seconds or more and get chunks using 
        # the imported function.
        # chunks = split_on_silence (
        #     # Use the loaded audio.
        #     audio_clip, 
        #     # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        #     min_silence_len = 2000,                 #EJD MAY NEED TO ADJUST!!!!
        #     # Consider a chunk silent if it's quieter than -16 dBFS.
        #     # (You may want to adjust this parameter.)
        #     silence_thresh = -25                #EJD MAY NEED TO ADJUST!!!!
        # )
        
        # # Process each chunk with your parameters
        # for i, chunk in enumerate(chunks):
        #     # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        #     silence_chunk = AudioSegment.silent(duration=500)
        
        #     # Add the padding chunk to beginning and end of the entire chunk.
        #     audio_chunk = silence_chunk + chunk + silence_chunk
        
        #     # Normalize the entire chunk.
        #     normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
            
        #     #EJD ADDITION : REMOVE PADDING FROM CHUNK
        #     #chunk_len = get_audio_len(normalized_chunk)
        #     #normalized_chunk = normalized_chunk[500:((chunk_len*1000)-500)]
            
        
        #     # Export the audio chunk with new bitrate.
        #     print("Exporting chunk{0}.wav.".format(i))
        #     normalized_chunk.export(
        #         ".//chunk{0}.wav".format(i),
        #         bitrate = "192k",
        #         format = "wav"
        #     )
        
         
        #%% FEED CHUNKS THROUGH PYTORCH SPEECH RECOGNITION ROUTINE
        print("*****EXTRACTING TEXT FROM AUDIO WITH PYTORCHAUDIO PIPELINE*****")
        
        # get list of audio chunks containing tag
        tag = "*chunk*"
        file_list = glob.glob(tag)
        file_list.sort() # make sure they are sorted in ascending order
        n_files = len(file_list)
        
        
        # begin_time = time.perf_counter()
        
        # Run each .WAV audio file chunk through speech2text voice recognition pipeline to
        # extract text from audio
        
        for files in file_list:
            
            import call_to_speech2text_al
            base_str = os.path.splitext(files)[0]
            txtFileOut = base_str+"_"+"txt_frm_spch.txt"
            txtFileOut_al = base_str+"_"+"txt_frm_spch_al.txt"
            speech_out_dir = "_assets/"
            speech_out_dir = aud_chunks_dir+speech_out_dir
            
            AudioIn = files
            data_dir = aud_chunks_dir
            out_dir = speech_out_dir
            
            #data_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/CHA_VIDEO/test_trimmed/out_dir/audio_speech_dta/audio_chunks/"
            #out_dir = '/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/CHA_VIDEO/test_trimmed/out_dir/audio_speech_dta/audio_chunks/_assets/'
            #txtFileOut = "text_out.txt"
            #txtFileOut_al = "text_out_al.txt"
            #AudioIn = "000000000_test4_trimmed_chunk_spad_500_norm_20_0_to_10473.wav"
    
            call_to_speech2text_al.call_speech2txt(data_dir, out_dir, txtFileOut, txtFileOut_al, AudioIn)
            
            
        
            #speech2text_al.speech_rec_al(files, txtFileOut, txtFileOut_al, aud_chunks_dir, speech_out_dir)
            
    
            #txtFileOut = repr(txtFileOut)
            #txtFileOut_al = repr(txtFileOut_al)
    
        
        
        # end_time = time.perf_counter()
        # time_elapsed = end_time - begin_time
        # time_elapsed = str(time_elapsed)
        # end_time_message = "TOOK"+" "+time_elapsed+" "+"SECONDS"
        # print(end_time_message)
        # print(" ")
        # junk = "jnk"  
    
#%% TEST CALL

#vid_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/Course_Materials_Data/CHA/CHA1-2_VIDEO/test_trimmed/";
vid_dir = "/home/eduwell/python_projects/data_scraping/frmMac/Video_Text_Extraction/test_trimmed/"
# NOTE : ALWAYS END PATH STRINGS FOR DIRECTORIES WITH A "/".. had to chose a convention for concatenating stuff...
#video = "CHA_I_Introduction_to_the_Nervous_System_default.mp4";
video = "tst_trimmed2.mp4";
frame_dsrate = 10; # Specifies the frame rate at which video is initially sampled
cor_thr = 0.85; # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
detector='PANet_IC15'; # specifies detector to be used used by mmocr to detect/put bounding boxes around text
#detector='TextSnake'
recognizer='SAR'; # specifies the recognizer which interprets/"reads" the text image within bounding boxes

x_merge = 65; # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box.
ClustThr_factor = 3; # controls agglomerative clustering threshold distance (in units of n*average textbox height)


extract_all(vid_dir, video, frame_dsrate, cor_thr, detector, recognizer, x_merge, ClustThr_factor)

