#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 20:03:12 2022

Note: had issue with not being able to find/use cuda despite it being installed etc.. torch couldn't get at it.. solved using this:
    https://askubuntu.com/questions/1330041/runtimeerror-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set-up-envi
    sudo rmmod nvidia_uvm
    sudo rmmod nvidia
    sudo modprobe nvidia
    sudo modprobe nvidia_uvm   

to test if torch can find cuda, use this:
    torch.cuda.is_available()
    should return "true" if it can
    
To see the processes running on the gpu to verify a job is in fact using it/running on it type "nvidia-smi" on the unix command line..
to clear all the space that can be freed up on the gpu using torch package in python type: torch.cuda.empty_cache()
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
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
import shutil
#from pydub import AudioSegment
#from pydub.silence import split_on_silence
import wave
import contextlib
#from scipy.io.wavfile import read
from scipy import stats
import peakutils

import pdfplumber
#from gingerit.gingerit import GingerIt
import copy
import pandas as pd
#import whisper
import torch


#%% Parameters
#config_dir = "/scratch/g/tark/dataScraping/envs/ocr/env/configs" #location of configs directory.. (may need to update if installing in new location..)
#config_dir = "/scratch/g/tark/installTesting/dataScraping/install/ocr/configs"
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
def extract_all(vid_dir, video, frame_dsrate, cor_thr, detector, recognizer, x_merge, ClustThr_factor,det_ckpt_in,recog_ckpt_in,config_dir):
    # Clear GPU cache..
    torch.cuda.empty_cache()
    audio_db = 0;
    
    # Go to video directory
    os.chdir(vid_dir)
    #%% 1) Set up the local directory structure..
    print("")
    print("**********************************************************")
    print("****************  SETTING UP DIRECTORIES  ****************")
    print("**********************************************************")
    print("")
    out_dir = "out_dir"
    # Set up output directory structure and copy video file in .. 
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
    
    #%% 2) Extract the audio and image frame data from the video file .. 
    print("**********************************************************")
    print("***********  EXTRACTING AUDIO FROM VIDEO FILE  ***********") 
    print("**********************************************************")
    print("")
    # Extract .WAV audio from video and save in audio directory
    clip = mp.VideoFileClip(video) 
    audioOut_fname = os.path.splitext(video)[0]+".wav" 
    
    # enter audio directory, convert video to audio, and save as .wav file
    os.chdir(vid_spdir)
    #clip.audio.write_audiofile(audioOut_fname)
    clip.audio.to_audiofile(audioOut_fname)
    os.chdir(root_dir)
    
    print("**********************************************************")
    print("**********  EXTRACTING IMAGE FRAMES FROM VIDEO  **********")
    print("**********************************************************")
    
    # Convert video into image frames
    ds = 1
    begin_time = time.perf_counter()
    img2txt.vid2frms(root_dir,video,vid_imdir,frame_dsrate,ds)
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "EXTRACTING IMAGE FRAMES TOOK"+" "+time_elapsed+" "+"SECONDS"
    print(end_time_message)
    print(" ")
    
    #%% 3) Reduce the number of video image frames to minimum number of "unique" frames...
    if audio_db ==0: # added this if statement to be able to bypass this stage when debugging the audio..
        
        print("**********************************************************")
        print("*********  FINDING MINIMUM SET OF UNIQUE FRAMES  *********")
        print("**********************************************************")
        print("")
        
        begin_time = time.perf_counter()
        # Reduce image frame set to minimum number of unique frames
        # Save unique frames in new directory
        uf_outdir = "unique_frames"
        tag = "*frame_*";
        #uIm_strtEndtimes = img2txt.unique_frms(vid_imdir,uf_outdir, tag, cor_thr,frame_dsrate) #ORIG
        uIm_strtEndtimes = img2txt.unique_frms2(vid_imdir,uf_outdir, tag, cor_thr,frame_dsrate) #EJD 3_30_23 update to speed up correlation step..
        end_time = time.perf_counter()
        time_elapsed = end_time - begin_time
        time_elapsed = str(time_elapsed)
        end_time_message = "GETTING UNIQUE FRAMES TOOK"+" "+time_elapsed+" "+"SECONDS"
        print(end_time_message)
        print(" ")
   #%% 4) Run the unique image frames through mmocr text detection pipeline to detext/extract text present in each image..  
        # Copy in the "configs" directory for mmocr
        source_config = config_dir
        
        destination_config = vid_imdir+"unique_frames/configs"
        if not os.path.exists(destination_config):
            shutil.copytree(source_config, destination_config)
        
        print("**********************************************************")
        print("*****  EXTRACTING TEXT FROM IMAGE FRAMES WITH MMOCR  *****")
        print("**********************************************************")
        print("")
        begin_time = time.perf_counter()
        # run individual frames through mmocr to extract text
        os.chdir(vid_imdir+uf_outdir)
        mmocr_dir = "mmocr_out/"
        if not os.path.exists(mmocr_dir):
            os.makedirs(mmocr_dir)
        
        # empty the gpu cache
        torch.cuda.empty_cache()
        #Run each cleaned frame through the mmocr text detection algoritm to detect and bound text
        frm2txt_DL2.frm2txt_mmocr_det(vid_imdir+uf_outdir,'*_sec_mean.jpeg*',mmocr_dir, detector,recognizer,x_merge,det_ckpt_in,recog_ckpt_in)
        # empty the gpu cache
        torch.cuda.empty_cache()

        # Run agglomerative clustering on the text output bounding boxes from mmocr to cluster text in a human readable order to then read out into a .txt file
        txt_out = frm2txt_DL2.bbox_txtMask_snek(vid_imdir+uf_outdir+"/"+mmocr_dir, vid_imdir+uf_outdir+"/"+mmocr_dir, "*.png*","*_sec_mean.json*", ClustThr_factor, "euclidean", "ward")
        
        
        txt_outname = os.path.splitext(video)[0]+"_framesText.txt"
        txt_outname_fxr = os.path.splitext(video)[0]+"_framesText_ocrFixr.txt"
        
        txtDtaTime = np.zeros((len(txt_out),3)).astype('object') # initialize dataframe for
        txtDtaTime_df = pd.DataFrame(copy.deepcopy(txtDtaTime), columns = ["startTime","endTime","txtDta"])
        
        
        print("Saving the text scraped from the "+str(len(txt_out))+" unique image frames to file:")
        frame_itr = 0;
        with open(txt_outname, 'w') as f:
            for frames in txt_out:
                frame_str = "TEXT SCRAPED FROM UNIQUE FRAME #"+" "+str(frame_itr).zfill(4)
                strz = "****************************************************************************************"
                
                # print(strz)
                # print('\n')
                #print("Saving text scraped from unique frame #"+str(frame_itr).zfill(4)+" ...")
                # print('\n')
                # print(strz)
                # print('\n')
                # print(strz)
                # print('\n')
                
                
                f.write(strz)
                print(strz)
                f.write('\n')
                f.write(strz)
                print(strz)
                f.write('\n')
                f.write(frame_str)
                print(frame_str)
                f.write('\n')
                f.write(strz)
                print(strz)
                f.write('\n')
                f.write(strz)
                print(strz)
                print("")
                f.write('\n')
                f.write('\n')
                
                # Save frame start end times and cluster list in txtDtaTime
                #txtDtaTime[frame_itr,0] = uIm_strtEndtimes[frame_itr][0] #grab the start time from this frame.
                #txtDtaTime[frame_itr,1] = uIm_strtEndtimes[frame_itr][1] #grab the end time from this frame.
                #txtDtaTime[frame_itr,2] = frames #grab the text structure from this frame.
                #["startTime","endTime","txtDta"]
                txtDtaTime_df.startTime[frame_itr] = uIm_strtEndtimes[frame_itr][0] #grab the start time from this frame.
                txtDtaTime_df.endTime[frame_itr] = uIm_strtEndtimes[frame_itr][1] #grab the end time from this frame.
                txtDtaTime_df.txtDta[frame_itr] = frames #grab the text structure from this frame.
                
                # Loop through txt_out and print contained text into output .txt file
                cntr = 0
                for txt_clusters in frames:
                    for sub_clustrs in txt_clusters:
                        if cntr > 0:
                            f.write('\n')
                            print("")
                        for txt_strgs in sub_clustrs:
                            txt2read = txt_strgs
                            f.write(txt2read)
                            print(txt2read)
                            f.write('\n')
                    f.write('\n')
                    f.write('\n')
                    print("")
                    
                    cntr = cntr+1
                frame_itr = frame_itr+1
        
           
        f.close()
        pause = ""
        
        # Go to video directory and save txtDtaTime as a .csv
        os.chdir(vid_imdir)
        ufCsv_outname = os.path.splitext(video)[0]+"_ufTxt-Time.csv" # make unique file name
        txtDtaTime_df.to_csv(ufCsv_outname)  
        #np.savetxt(ufCsv_outname, txtDtaTime, delimiter=',') # save as .csv
        
        os.chdir(root_dir)
        end_time = time.perf_counter()
        time_elapsed = end_time - begin_time
        time_elapsed = str(time_elapsed)
        
        print(strz)
        print(strz)
        print("")
        
        end_time_message = "MMOCR TEXT DETECTION/EXTRACTION TOOK"+" "+time_elapsed+" "+"SECONDS"
        print(end_time_message)
        print(" ")
        
        # Clear GPU cache..
        torch.cuda.empty_cache()
    #%% Detect, recognize, and extract spoken text in audio file using Whisper
    #EJD ADDED THE IF DB_SKIP STATEMENT BELOW TO HAVE THE MMOCR ENV SKIP OVER THE WHISPER STUFF FOR NOW..
    if db_skip == 0:
        print("*****DETECTING-RECOGNIZING-AND-EXTRACTING TEXT FROM AUDIO VIA WHISPER*****")
        txt_outname_aud = os.path.splitext(video)[0]+"_AudioTextWhspr.txt"
        begin_time = time.perf_counter()
        # go to the audio directory..
        os.chdir(vid_spdir)
        # Load Whisper Model:
        #--------------------------------------------------------------------------
    
        mdl_card = "medium"
        
        # 1.. The simple way..
        # --------------------------------------------------------------
        # empty the gpu cache
        torch.cuda.empty_cache()
        
        model = whisper.load_model(mdl_card)
        
        # Run Model on Audio
        #whspr_result = model.transcribe(audioOut_fname) # orig, no options specified...
        whspr_result = model.transcribe(audioOut_fname, without_timestamps=False, max_initial_timestamp=None) 
        
        # empty the gpu cache
        torch.cuda.empty_cache()
        # --------------------------------------------------------------
        
        # 2.. The more complicated way specifying language detection, etc..
        # As of now (12/30/22), I still haven't gotten this kind of more "granular"
        # implimentation to work yet... works when I tun the simple/standard way above
        # presumably because the default options include important stuff I'm not 
        # addressing/don't know to specify in the custom call below..
        #
        # --------------------------------------------------------------
        # model = whisper.load_model(mdl_card)
    
        # # load audio and pad/trim it to fit 30 seconds
        # audio = whisper.load_audio(audioOut_fname)
        # audio = whisper.pad_or_trim(audio)
        
        # # make log-Mel spectrogram and move to the same device as the model
        # mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # #detect the spoken language
        # _, probs = model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")
        
        # # decode the audio
        # options = whisper.DecodingOptions(language="en") #ejd added language="en" to force english.. 
        # result = whisper.decode(model, mel, options)
        
        # --------------------------------------------------------------   
        
        
        
        print(" ")
        print("Text extracted from audio via Whisper:")
        #print(result.text)
        print(whspr_result["text"])
        
        with open(txt_outname_aud, 'w') as f:
            f.write(whspr_result["text"])
        f.close()
        
        end_time = time.perf_counter()
        time_elapsed = end_time - begin_time
        time_elapsed = str(time_elapsed)
        print(" ")
        end_time_message = "RUNNING WHISPER TOOK"+" "+time_elapsed+" "+"SECONDS"+" ("+mdl_card+" model)"
        print(end_time_message)
        print(" ") 
        #--------------------------------------------------------------------------
        
        #%% Combine text data scraped from the video image frames and audio into an output data file along with timing information
        
        # List of relavent data we want to include and notes on their organization:
        # -------------------------------------------------------------------------
        # (1) Text extracted from each of the unique image frames: currenltly stored in "txt_out"
        #    (sry for the lengthy description.. but if you need to understand the contents of txt_out, you'll need to understand all this..)
        #       * txt_out is a list of nested lists
        #       * each of the "highest level" lists is a video frame. (lets call these "frame_lists")
        #       * each frame_list is also a "list of nested lists" containing 1 or more nested lists.
        #       * each of the nested lists within a frame_list is a cluster-of-clusters of text fields.
        #           o these "clusters-of-clusters" essentially amount to a 'vertical text projection column' on the frame..
        #           o essentially, if you were to "flatten" the vertical y-dimension of the page, these would
        #             be text fields which would squish on top of one another in the x dimension..
        #           o another analogy which might help: if text clusters were playdoh balls and you were to
        #             connect all playdoh balls with those falling directly below/above them (ones sharing  
        #             x-coordinates..), you would get the "clusters-of-clusters" referenced above. Hopefully 
        #             this helps.. Ethan wrote this clustering algorithm.. it may be a re-invention/re-expression
        #             of an existing one, but it was what he could come up with at the time to order local clusters
        #             of text detected by mmocr/clustered using agglomerative clustering in a "human readable".. see 
        #             frm2txt_DL2.bbox_txtMask_snek ...
        #       * each of the "clusters-of-clusters" described above is composed of localized clusters of text fields created using
        #         Agglomerative clustering.. Ethan had originally hoped that Agglomerative clustering would solve all
        #         his text organization/clustering/sequencing woes.. but then discovered he still had the same issue he began with prior
        #         to agglomerative clustering.. only before the question was "how can I use the bounding box coordinates to read these various text fields
        #         (from mmocr detection) out into a text file in the order you'd read them off the frame (sequence in a human readable manner)..
        #         agglomerative clustering was good for clustering localized bits of text which should be read together.. but we still had the same
        #         issue afterwards.. only now it was for clusters... hence the clusters-of-clusters described above. Aaaaanyyyhoo.. point of the matter
        #         is that the "clusters" which make up the "clusters-of-clusters" are (spatial) clusters of individual text field outputs from
        #         mmocr, clustered via agglomerative clustering. Each of these is a single list of text strings. (yay!! a list! not a list-of-lists! or a list-of-lists-of-lists!..)
        #       * Finally, whats inside the clusters described above? Welp, as mentioned to above, each cluster is saved as a list of strings. Each of those strings
        #         is the text contents of a single bounding box of text extracted by mmocr/saved in the output .json ...
        #           o NOTE: within each of these strings, there MAY have also been grouping done by MMOCR.. the grouping I speak of pertains to
        #             the "x_merge" option specified when calling mmocr.. this basically specifies how far laterally (in the x direction) mmocr will look to merge text boxes into the same box.
        #             so.. even the individual strings may have been seperate bits of text initially detected, but then merged via x-merge..            
        #
        # (2) The "start" and "end" times for each of the unique image frames: currenltly stored in "uIm_strtEndtimes"
        #       * uIm_strtEndtimes is a simple list-of-lists
        #       * each list corresponds to a unique frame and contains 2 numbers index [0]: the start time, index [1]: the end time
        #       * because the "unique frames" are the average of series of images deemed "the same" by the cross correlation method 
        #         (ie.. the frame-to-frame image correlation in the series remains above the selected correlation threshold), the start
        #         and end times should correspond to major changes happening on screen.. in our case with video lectures, hopefully the 
        #         start and end of individual lecture slides presented...
        #
        # (3) Specific contents from each of the "segments" within the output from Whisper (audio speech detection): currently stored in "whspr_result"
        #       * at minimum, this should hopefully include: 
        #            a. the start and end time of the segment relative to the beginning of the recording
        #            b. the text string
        
        
        
        # Get the text segment list of dictionaries..
        whspr_result_txtSegs = whspr_result["segments"]
        n_txtSegs = len(whspr_result_txtSegs)
        
        # STOPPING HERE FOR THE NIGHT (12/30/22, 7:14 PM...)
        # NEXT TIME, WE WANT TO:
        #
        # Make a dataframe that is n_txtSegs rows long and as many cols wide as things we want to include..
        #       At the moment, I'm thinking:
            #       (From whspr_result_txtSegs)
            #       text
            #       start
            #       end
            #       no_speech_prob
            #
            #       (From txt_out)
            #       ?everything as is?
            #
            #       uIm_strtEndtimes (everything as is??)
        
       
        txtDta_all = np.zeros((n_txtSegs,4)).astype('object') # initialize dataframe
        
        
        pause = "";
        countr = 0;
        for segs in whspr_result_txtSegs:
            txtDta_all[countr][0] = whspr_result_txtSegs[countr]["start"]
            txtDta_all[countr][1] = whspr_result_txtSegs[countr]["end"]
            txtDta_all[countr][2] = whspr_result_txtSegs[countr]["text"]
            txtDta_all[countr][3] = whspr_result_txtSegs[countr]["no_speech_prob"]
            
            
            countr = countr+1;
        del countr
        txtDta_all_df = pd.DataFrame(copy.deepcopy(txtDta_all), columns = ["whspr_start","whspr_end","whspr_text","whspr_no_speech_prob"])
        #txtDta_all = txtDta_all.rename(columns={"0":"start","1":"end","2":"text","3":"no_speech_prob"}) #rename columns..
        
        #Need to add txt data from image frames separately..     
        #txtDta_all[countr][5] =txt_out[countr]
        #txtDta_all[countr][6] =uIm_strtEndtimes[countr]
        countr = 0;
        tmp_df = copy.deepcopy(txtDta_all_df.whspr_start)
        tmp_df = tmp_df.rename("UnqImgFrm_txt")
        tmp_df2 = copy.deepcopy(txtDta_all_df.whspr_start)
        tmp_df2 = tmp_df2.rename("UnqImgFrm_Strt")
        tmp_df3 = copy.deepcopy(txtDta_all_df.whspr_end)
        tmp_df3 = tmp_df3.rename("UnqImgFrm_Stop")
        unqFrm_numCol = np.zeros((n_txtSegs,1)).astype('object') # initialize dataframe
        unqFrm_numCol = pd.DataFrame(copy.deepcopy(unqFrm_numCol), columns = ["UnqImgFrm_num"]) #rename
        unqFrm_numCol.UnqImgFrm_num.replace(0, np.nan, inplace=True) # replace the 0s with nans
        for frms in txt_out:        
            for zz in range(0,tmp_df.shape[0]):
                if (txtDta_all_df.whspr_start[zz] >= uIm_strtEndtimes[countr][0]) and (txtDta_all_df.whspr_start[zz] <= uIm_strtEndtimes[countr][1]):
                    tmp_df[zz] = frms
                    tmp_df2[zz] = uIm_strtEndtimes[countr][0]
                    tmp_df3[zz] = uIm_strtEndtimes[countr][1]
                    unqFrm_numCol.UnqImgFrm_num[zz] = countr
                
            
                    
            countr=countr+1
        del countr
        txtDta_all_df = pd.concat([txtDta_all_df,tmp_df],axis=1)
        txtDta_all_df = pd.concat([txtDta_all_df,tmp_df2],axis=1)
        txtDta_all_df = pd.concat([txtDta_all_df,tmp_df3],axis=1)
        txtDta_all_df = pd.concat([txtDta_all_df,unqFrm_numCol.UnqImgFrm_num],axis=1)
        
        
        
        txtDta_all_df.to_csv(root_dir+"/"+"imgFrm_audio_text_combined_out.csv")            
        # Clear the GPU Cache (make sure this is called as the final command..)
        torch.cuda.empty_cache()
        #%% OLD AUDIO ATTEMPT ... 
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
    
        
            # OLD METHOD OF SILENCE SPLITTING.. (EJD COMMENTED OUT BC HE WROTE HIS OWN ABOVE)
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
            
             
            #FEED CHUNKS THROUGH PYTORCH SPEECH RECOGNITION ROUTINE
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
    
#%% GET INPUT PARAMETERS AND RUN PIPELINE..

# Get Parameters .. 
#vid_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/Course_Materials_Data/CHA/CHA1-2_VIDEO/test_trimmed/";
#vid_dir = "/scratch/g/tark/dataScraping/envs/ocr/env/video_test/";
vid_dir = sys.argv[1];

# NOTE : ALWAYS END PATH STRINGS FOR DIRECTORIES WITH A "/".. had to chose a convention for concatenating stuff...
#video = "CHA_I_Introduction_to_the_Nervous_System_default.mp4";
#video = "Vasculature of the Abdomen_default.mp4";
video = sys.argv[2];

#frame_dsrate = 15; # Specifies the frame rate at which video is initially sampled
frame_dsrate = int(sys.argv[3]); # Specifies the frame rate at which video is initially sampled

#cor_thr = 0.95; # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"
cor_thr = float(sys.argv[4]); # Controls the correlation threshold used to determine when enough has changed in video to count as a "new unique frame"

#detector='PANet_IC15'; # specifies detector to be used used by mmocr to detect/put bounding boxes around text
detector=sys.argv[5]; # specifies detector to be used used by mmocr to detect/put bounding boxes around text

#detector='TextSnake'
#recognizer='SAR'; # specifies the recognizer which interprets/"reads" the text image within bounding boxes
recognizer=sys.argv[6]; # specifies the recognizer which interprets/"reads" the text image within bounding boxes

#x_merge = 65; # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box.
x_merge = int(sys.argv[7]); # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box.

#ClustThr_factor = 3; # controls agglomerative clustering threshold distance (in units of n*average textbox height)
ClustThr_factor = int(sys.argv[8]); # controls agglomerative clustering threshold distance (in units of n*average textbox height)

#det_ckpt_in='/scratch/g/tark/dataScraping/envs/ocr/env/mmocrChkpts/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth';
det_ckpt_in=sys.argv[9];

#recog_ckpt_in='/scratch/g/tark/dataScraping/envs/ocr/env/mmocrChkpts/sar_r31_parallel_decoder_academic-dba3a4a3.pth';
recog_ckpt_in=sys.argv[10];

config_dir=sys.argv[11];

# Run the pipeline .. 
torch.cuda.empty_cache()
extract_all(vid_dir, video, frame_dsrate, cor_thr, detector, recognizer, x_merge, ClustThr_factor,det_ckpt_in,recog_ckpt_in,config_dir)
torch.cuda.empty_cache()

