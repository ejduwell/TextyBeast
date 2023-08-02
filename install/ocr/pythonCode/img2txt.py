#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 19 14:26:55 2022

https://nanonets.com/blog/ocr-with-tesseract/
https://docs.opencv.org/4.x/d1/dfb/intro.html
https://pypi.org/project/pytesseract/

@author: eduwell
"""

#%% vid2frms
def vid2frms(work_dir,video_in,out_dir,frameRate,ds):
    #%% Import Packages
    import os
    import cv2
    import moviepy.editor
    #%% Parameters
    if ds == 0:
        no_ds = 1
    if ds == 1:
        no_ds = 0
    #%% Read in the Video
    
    # go to your working directory where the video is located
    os.chdir(work_dir)
    
    # read in the video
    vidcap = cv2.VideoCapture(video_in)
    video = moviepy.editor.VideoFileClip(video_in)
    # create and enter the output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)

    #%% Extract the video frames as images
    
    # if downsampling is requested, extract frames at specified rate
    if no_ds == 0:
        
        #vid_fpath = work_dir+video_in
        
        video_duration = int(video.duration)
        #video_duration = video_duration*
        pause = ""
        def getFrame(sec): 
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
            hasFrames,image = vidcap.read() 
            if hasFrames: 
                cv2.imwrite("frame_"+str(sec).zfill(5)+"_sec.JPEG", image)     # save frame as JPEG file 
                return hasFrames 
        
        # initialize sec as 0
        sec = 0 
    
        success = getFrame(sec) 
        while success and (sec < (video_duration+1)): 
            sec = sec + frameRate 
            sec = round(sec, 2) 
            success = getFrame(sec)
    
    # if no downsampling is requested, extract all the frames
    if no_ds == 1:
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("frame%d.JPEG" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print("Writing frame%d.. Does it exist?: " % count, success)
            count += 1 
    # go back to the starting work directory
    os.chdir(work_dir)

#%% unique_frames module
# The purpose of this module is to eliminate redundant frames and reduce the 
# number of frames extracted from video to a minimum number of quality images
# containing all of the available text.

def unique_frms(work_dir, out_dir, tag, cor_thr,frame_dsrate):
    # add d_thr input param..

    #%% Import Packages
    import os
    import cv2
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import diff
    import time
    #import pandas as pd
    #from scipy import signal
    #%% enter working directory and setup file list
    strt_dir = os.getcwd() 
    os.chdir(work_dir)
    
    # make output directory if it doesnt already exist.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # get list of images containing tag
    file_list = glob.glob(tag)
    file_list.sort() # make sure they are sorted in ascending order
    n_files = len(file_list)
    
    #%% generate corrleation timeseries for each frame
    
    # (cross correlate each frame with all others in the set to detect relevant
    # timepoints of changes in slides)
    
    # first initiallize a vector to store the images read in.
    # also initialize a matrix for the image correlation timeseries
    
    #import test image for dimensions
    img4dims = cv2.imread(file_list[0], 0)
    im_dims_2d = img4dims.shape
    #img4dims = cv2.normalize(img4dims, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img4dims = img4dims.flatten(order='C')
    [x] = img4dims.shape
    
    
    
    im_mat = np.empty(shape=(n_files,x),dtype='uint8')
    #frm_label_vect = np.empty(n_files);
    imcorr_mat = np.empty(shape=(n_files,n_files),dtype='float32')
    imcorr_mat_th = np.empty(shape=(n_files,n_files),dtype='uint8')
    #imcorr_mat_deriv = np.empty(shape=(n_files,n_files-1),dtype='float32')
    #change_ind_mat = np.empty(shape=(n_files,2),dtype='uint8')
    
    ii = -1; # initialize a number ii for keeping track of iterations
    # then read the image files into im_vect
    
    begin_time = time.perf_counter()
    print("reading in images ...")
    for file in file_list:
        ii = ii+1
        im_in = cv2.imread(file, 0);
        #im_in = cv2.normalize(im_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im_in = im_in.flatten(order='C')
        im_mat[ii][:] = im_in
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "reading in images took"+" "+time_elapsed+" "+"seconds"
    print("finished reading in images ...")
    print(end_time_message)    
    
    ii = 0 # reinitialize a number ii for keeping track of iterations
    oo = 0# initialize a number ii for keeping track of iterations
    
    begin_time = time.perf_counter()
    print('')
    print("creating correlation matrix ...")
    for ii in range(n_files):
        #ii = ii+1
        #oo = -1 # initialize a number ii for keeping track of iterations
        im1 = im_mat[ii][:]
        for oo in range(n_files):
            #oo = oo+1
            im2 = im_mat[oo][:]
            cor = np.corrcoef(im1, im2) 
            cor = cor.min()
            #cor = signal.correlate2d(im1, im2)
            imcorr_mat[ii,oo] = cor
    #plt.imshow(imcorr_mat,vmin=0, vmax=1.,cmap='plasma')
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "creating correlation matrix"+" "+time_elapsed+" "+"seconds"
    print('')
    print("finished creating correlation matrix ...")
    print('')
    print(end_time_message) 
    plt.imshow(imcorr_mat,vmin=0, vmax=1.,cmap='plasma')
    
    
    begin_time = time.perf_counter()
    print('')
    #cor_thr = 0.8  # EJD NOTE**** : try raising to 0.85 - 0.9....
    print("thresholding correlation matrix rows at th="+str(cor_thr)+" ...") 
    for ii in range(n_files):

        for oo in range(n_files):
            if imcorr_mat[ii][oo] >= cor_thr:
                imcorr_mat_th[ii][oo] = 1
            else:
                imcorr_mat_th[ii][oo] = 0
    
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "thresholding correlation matrix took"+" "+time_elapsed+" "+"seconds"
    print('')
    print("finished thresholding correlation matrix ...")
    print('')
    print(end_time_message) 
    plt.imshow(imcorr_mat_th,vmin=0, vmax=1.,cmap='gray')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    
    # Make an array where each row contains the file of adjacent "matching" 
    #image frames (temporally images with the same row vector in imcorr_mat_th) 
    unique_frm_indices_masterlist = [[file_list[0]]]
    #unique_frm_indices = [file_list[0]]
    oo = 0 # reinitiate oo as an arbitrary iterator variable..
    for ii in range(1,n_files):        
        if np.array_equal(imcorr_mat_th[ii][:], imcorr_mat_th[ii-1][:]):
                unique_frm_indices_masterlist[oo].append(file_list[ii])
        else:
            oo = oo+1
            unique_frm_indices_masterlist.append([file_list[ii]])
    junk = "jnk"
    
    # For each list of duplicate unique frames in unique_frm_indices_masterlist
    # re-read in the images, and average them to create a single mean image
    aveImg_strtEndtimes = []; # initialize
    for ii in unique_frm_indices_masterlist:
        image_list = ii
        #im_mat_temp = np.empty(shape=(im_dims_2d[0],im_dims_2d[1],3,len(image_list)),dtype='uint8')
        #n_im = len(image_list)
        #oo =0;
        
        image_data = []
        for oo in image_list:
            this_image = cv2.imread(oo, 1)
            image_data.append(this_image)
    
        avg_image = image_data[0]
        for i in range(len(image_data)):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
        
       
        import re

      
        pattern = "_(.*)_" # wildcard for finding numbers in the file name string (they will always be the only thing flanked by '_'s..)
        compiled = re.compile(pattern)

        
        
        
        first_im = os.path.splitext(image_list[0])[0]
        last_im  = os.path.splitext(image_list[-1])[0]
        
        last_im_num = compiled.search(last_im)
        last_im_num = int(last_im_num.group(1).strip())+frame_dsrate # EJD added + frame_dsrate to account for fact this is only image sample *up to* the next one
        
        first_im_num = compiled.search(first_im)
        first_im_num = int(first_im_num.group(1).strip())
        
        times = [first_im_num,last_im_num]
        aveImg_strtEndtimes.append(times)
        
        name_str = first_im+"_to_"+last_im+"_mean.jpeg"
        os.chdir(out_dir)
        cv2.imwrite(name_str, avg_image) 
        os.chdir(work_dir)
                
        # cv2.imwrite('avg_happy_face.png', avg_image)
        # for oo in range(0,n_im):
        #     im_mat_temp[:][:][oo] = cv2.imread(image_list[oo])
        # if len(image_list) > 1:    
        #     mean_array = np.mean(im_mat_temp[:][:][:], axis = 4)
        # else:
        #     mean_array = cv2.imread(image_list[oo])
        

    
    # # take the first derivative of each row in the imcorr matrix to express
    # # in terms of "change"
    # begin_time = time.perf_counter()
    # print('')
    # print("*****TAKING FIRST DERIVATIVE OF EACH ROW IN CORRELATION MATRIX*****")   
    # for ii in range(n_files):
    #     dx = 0.1
    #     row = imcorr_mat[ii][:]
    #     drow = diff(row)/dx
    #     imcorr_mat_deriv[ii][:] = np.abs(drow)
    # end_time = time.perf_counter()
    # time_elapsed = end_time - begin_time
    # time_elapsed = str(time_elapsed)
    # end_time_message = "TAKING FIRST DERIVATIVE OF EACH ROW IN CORRELATION MATRIX"+" "+time_elapsed+" "+"SECONDS"
    # print('')
    # print("*****FINISHED TAKING FIRST DERIVATIVE OF EACH ROW IN CORRELATION MATRIX*****")
    # print('')
    # print(end_time_message) 
    
    # plt.imshow(imcorr_mat_deriv,vmin=imcorr_mat_deriv.min(), vmax=imcorr_mat_deriv.max(),cmap='plasma')
    # # find two largest values (points with biggest change in correlation)in 
    # # each row of the derivative matrix to get beginning and end of high 
    # # correlation stretch in each row
    # begin_time = time.perf_counter()
    # print('')
    # print("*****LOCATING INDICES OF 2 LARGEST VALUES IN EACH ROW OF 1ST DERIVATIVE MATRIX*****")   
    # for ii in range(n_files):
    #     n=2
    #     row = imcorr_mat_deriv[ii][:]
    #     ind = np.argpartition(row, -n)[-n:]
    #     change_ind_mat[ii][:] = ind
    # end_time = time.perf_counter()
    # time_elapsed = end_time - begin_time
    # time_elapsed = str(time_elapsed)
    # end_time_message = "LOCATING INDICES OF 2 LARGEST VALUES IN EACH ROW OF 1ST DERIVATIVE MATRIX"+" "+time_elapsed+" "+"SECONDS"
    # print('')
    # print("*****LOCATING INDICES OF 2 LARGEST VALUES IN EACH ROW OF 1ST DERIVATIVE MATRIX*****")
    # print('')
    # print(end_time_message) 
    
    
    # re order each row of change_ind_mat so they're ordered from low index to high
    
    os.chdir(strt_dir)
    return aveImg_strtEndtimes
#%% ts_pipeline module

def unique_frms2(work_dir, out_dir, tag, cor_thr,frame_dsrate):
    # add d_thr input param..

    #%% Import Packages
    import os
    import cv2
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import diff
    import time
    #import pandas as pd
    #from scipy import signal
    #%% enter working directory and setup file list
    strt_dir = os.getcwd() 
    os.chdir(work_dir)
    
    # make output directory if it doesnt already exist.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # get list of images containing tag
    file_list = glob.glob(tag)
    file_list.sort() # make sure they are sorted in ascending order
    n_files = len(file_list)
    
    #import test image for dimensions
    img4dims = cv2.imread(file_list[0], 0)
    im_dims_2d = img4dims.shape
    #img4dims = cv2.normalize(img4dims, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img4dims = img4dims.flatten(order='C')
    [x] = img4dims.shape
    
    im_mat = np.empty(shape=(n_files,x),dtype='uint8')
    im_mat_corBin = np.empty(shape=(n_files,1),dtype='uint8') # initialize array for storing 0/1s for binary indication of whether threshold has been met..
    
    #frm_label_vect = np.empty(n_files);
    imcorr_mat = np.empty(shape=(n_files,n_files),dtype='float32')
    imcorr_mat_th = np.empty(shape=(n_files,n_files),dtype='uint8')
    #imcorr_mat_deriv = np.empty(shape=(n_files,n_files-1),dtype='float32')
    #change_ind_mat = np.empty(shape=(n_files,2),dtype='uint8')
    
    ii = -1; # initialize a number ii for keeping track of iterations
    # then read the image files into im_vect
    
    begin_time = time.perf_counter()
    print("reading in images ...")
    for file in file_list:
        ii = ii+1
        im_in = cv2.imread(file, 0);
        #im_in = cv2.normalize(im_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im_in = im_in.flatten(order='C')
        im_mat[ii][:] = im_in
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "reading in images took"+" "+time_elapsed+" "+"seconds"
    print("finished reading in images ...")
    print(end_time_message)    
    
    ii = 0 # reinitialize a number ii for keeping track of iterations
    oo = 0# initialize a number ii for keeping track of iterations
    
    # Correlate each flattened frame with its immediate neighbor looking for locations where the next frame's
    # correlation with the current frame is lower than the threshold value
    # Mark these with 1s to indicate a transition point/big change
    # Mark all others with 0s.. (this is way faster than doing a full cross correlation ...)
    print('')
    print("creating correlation matrix ...")
    begin_time = time.perf_counter()
    for ii in range(n_files):
        if ii<n_files-1:
            im1 = im_mat[ii][:]
            im2 = im_mat[ii+1][:]
            cor = np.corrcoef(im1, im2) 
            cor = cor.min()
            if cor<=cor_thr:
                im_mat_corBin[ii][0] = 1; # mark with a 1 cor is less than or equal to threshold value..
            else:
                im_mat_corBin[ii][0] = 0; # mark with a 0 cor is not less than or equal to threshold value..
        else:
            im_mat_corBin[ii][0] = 1; # This is the last frame. Mark it with a 1 as it is a cut point by default..
                
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "creating correlation matrix took"+" "+time_elapsed+" "+"seconds"
    print('')
    print("finished creating correlation matrix ...")
    print('')
    print(end_time_message) 
    
    # Get the indices marked with 1s (location of changes exceeding thr..)
    im_mat_corBin_idxs = np.where(np.any(im_mat_corBin>0, axis=1))
    im_mat_corBin_idxs = im_mat_corBin_idxs[0]
   
    del ii
    countr = 0 #initialize countr
    aveImg_strtEndtimes = []; # initialize
    for ii in range(0,len(im_mat_corBin_idxs)):
        vals = im_mat_corBin_idxs[ii]
        if countr == 0:
            filz = file_list[0:vals+1]
        else:
            filz = file_list[im_mat_corBin_idxs[countr-1]+1:vals+1]
        
        image_list = filz
        
        #im_mat_temp = np.empty(shape=(im_dims_2d[0],im_dims_2d[1],3,len(image_list)),dtype='uint8')
        #n_im = len(image_list)
        #oo =0;
        
        image_data = []
        for oo in image_list:
            this_image = cv2.imread(oo, 1)
            image_data.append(this_image)
    
        avg_image = image_data[0]
        for i in range(len(image_data)):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
        
       
        import re

      
        pattern = "_(.*)_" # wildcard for finding numbers in the file name string (they will always be the only thing flanked by '_'s..)
        compiled = re.compile(pattern)
        
        first_im = os.path.splitext(image_list[0])[0]
        last_im  = os.path.splitext(image_list[-1])[0]
        
        last_im_num = compiled.search(last_im)
        last_im_num = int(last_im_num.group(1).strip())+frame_dsrate # EJD added + frame_dsrate to account for fact this is only image sample *up to* the next one
        
        first_im_num = compiled.search(first_im)
        first_im_num = int(first_im_num.group(1).strip())
        
        times = [first_im_num,last_im_num]
        aveImg_strtEndtimes.append(times)
        
        name_str = first_im+"_to_"+last_im+"_mean.jpeg"
        os.chdir(out_dir)
        cv2.imwrite(name_str, avg_image) 
        os.chdir(work_dir)
        
        countr = countr+1
    del countr
    
    os.chdir(strt_dir)
    return aveImg_strtEndtimes

def ts_pipeline(image_dir, image_in, out_dir, s_pstage, p_pstage):
    # SYNTAX: ts_pipeline("path/to/dir/with/images", "filename.JPEG", "desired_output_dir_name", number_0_to_2)
    # 
    # INPUT PARAMS:
    ##########################################################################
    # image_dir: directory containing the images from which you want to extract
    # text. (text string of path)
    #
    # image_in: the specific image file you want extract text from (file 
    # name text string)
    #
    # out_dir: the name of the output directory in which text and images will 
    # outputs will be saved (created by program.. text string)
    #
    # s_pstage: specifies whether you want to save the images created in 
    # intermediate pre-processing steps. 1=save intermediate images, 0=don't
    #
    # p_pstage: specifies whether you want to pass the intermediate processing
    # images as outputs 1=yes, 0=no
    #
    # OUTPUTS:
    ###########################################################################
    # text_out: text file containing text extracted from image
    #
    # final_im: Final image after preprocessng with bounding boxes around 
    # where Tesseract located text
    #
    # ind_ps: vector of all images from processing pipeline including the 
    # input image, all intermediate processing images, and the final image fed
    # into tesseract with bounding boxes around text.
    #
    #
    #
    #
    #
    #%% Import Packages
    import cv2 
    import pytesseract
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from pytesseract import Output
    #from scipy.ndimage import interpolation as inter
    #from IPython.display import Image
    #import sys
    #from PIL import Image as im
    
    #%% Parameters
    #image_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/video_clip_FRAMES"
    #image_in = "frame_519_sec.JPEG"
    out_file_tag = "_tesrc_text"
    #out_dir = "output_directory"
    #out_im = "final_image.JPEG"
    #s_pstage = 1;
    plot_graphs = 0;
    #%% Read in Image
    # go to image directory
    os.chdir(image_dir)
    
    img = cv2.imread(image_in, cv2.IMREAD_COLOR); #READ IN IMAGE AS color image
    
    #img_gry = cv2.imread(image_in, cv2.IMREAD_GRAYSCALE); #READ IN Gray copy too
    
    # create and enter the output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    os.chdir(out_dir)
    
    if s_pstage >= 1:
        str_000="ps000_orig_"+image_in 
        cv2.imwrite(str_000, img)
    
    #change color channels to be rgb (not cv2's weird bgr..)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    
    
    # get image dimensions
    #dimensions = img_rgb.shape;
    
    #%% Image Preprocessing Prior to OCR
        
    
    # Split the image into the B,G,R components
    img_r,img_g,img_b = cv2.split(img_rgb);
    
    if plot_graphs > 0:
        plt.figure(2)
        plt.figure(figsize=[50,10])
        plt.subplot(141);plt.imshow(img_r,cmap='gray');plt.title("Red Channel");
        plt.subplot(142);plt.imshow(img_g,cmap='gray');plt.title("Green Channel");
        plt.subplot(143);plt.imshow(img_b,cmap='gray');plt.title("Blue Channel");
        
        imgMerged = cv2.merge((img_b,img_g,img_r))
        plt.subplot(144);plt.imshow(imgMerged);plt.title("RGB Merged");
        plt.show(2)
        plt.close('all')
    
    
    if s_pstage > 1:
        str_0011="ps001_r_"+image_in
        cv2.imwrite(str_0011, img_r)
        str_0012="ps001_g_"+image_in
        cv2.imwrite(str_0012, img_g)
        str_0013="ps001_b_"+image_in 
        cv2.imwrite(str_0013, img_b)
    
    
    # Apply Adaptive Thesholding  
    #(EJDNOTE: may want to optimize threshold values later currently just copied from demo..)
    
    # make thresholded version of red channel
    img_r_adTh = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7);
    # make thresholded version of green channel
    img_g_adTh = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7);
    # make thresholded version of blue channel
    img_b_adTh = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7);
    
    # make thresholded version of greyscale version
    #img_gry_adTh = cv2.adaptiveThreshold(img_gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7);
    
    if plot_graphs > 0:
        plt.figure(3)
        plt.figure(figsize=[40,10])
        plt.subplot(345);plt.imshow(img_r_adTh,cmap='gray');plt.title("Red Channel w/ Adaptive Thresholding");
        plt.subplot(346);plt.imshow(img_g_adTh,cmap='gray');plt.title("Green Channel w/ Adaptive Thresholding");
        plt.subplot(347);plt.imshow(img_b_adTh,cmap='gray');plt.title("Blue Channel w/ Adaptive Thresholding");
        plt.show(3)
        plt.close('all')
    
    
    if s_pstage > 1:
        str_0021="ps002_r_aBT_"+image_in
        cv2.imwrite(str_0021, img_r_adTh)
        str_0022="ps002_g_aBT_"+image_in
        cv2.imwrite(str_0022, img_g_adTh)
        str_0023="ps002_b_aBT_"+image_in 
        cv2.imwrite(str_0023, img_b_adTh)
    
    # Combine the binary thesholded r,g,and b channel images logically using "or"
    img_rgb_adTh_com = cv2.bitwise_or(img_r_adTh, img_g_adTh, img_b_adTh, mask = None);
    
    if plot_graphs > 0:
        plt.figure(4)
        plt.figure(figsize=[40,10])
        plt.imshow(img_rgb_adTh_com,cmap='gray');plt.title("Thresholded R, B, and G Channels Combined Logically with OR");
        plt.show(4)
        plt.close('all')
    
    if s_pstage == 1:
        str_003="ps003_rgb_aBT_comb_"+image_in
        cv2.imwrite(str_003, img_rgb_adTh_com)
    
    
    # Skeletonization
    kernel1 = np.ones((2,2),np.uint8);
    kernel2 = np.ones((2,2),np.uint8);
    kernel3 = np.ones((2,2),np.uint8);
    kernel4 = np.ones((2,2),np.uint8);
    
    
    img_rgb_adTh_com_sk1 = cv2.erode(img_rgb_adTh_com,kernel1,iterations = 1);
    img_rgb_adTh_com_sk2 = cv2.erode(img_rgb_adTh_com,kernel2,iterations = 1);
    img_rgb_adTh_com_sk3 = cv2.erode(img_rgb_adTh_com,kernel3,iterations = 1);
    img_rgb_adTh_com_sk4 = cv2.erode(img_rgb_adTh_com,kernel4,iterations = 1);
    
    if plot_graphs > 0:
        plt.figure(5)
        plt.figure(figsize=[40,40])
        plt.imshow(img_rgb_adTh_com_sk1,cmap='gray');plt.title("Thresholded RGB Channels Combined Logically with OR and Skeletonized K1");
        plt.show(5)
        plt.close('all')
    
        plt.figure(6)
        plt.figure(figsize=[40,40])
        plt.imshow(img_rgb_adTh_com_sk2,cmap='gray');plt.title("Thresholded RGB Channels Combined Logically with OR and Skeletonized K2");
        plt.show(6)
        plt.close('all')
        
        # THIS KERNEL SIZE CURRENTLY LOOKS BEST...
        plt.figure(7)
        plt.figure(figsize=[40,40])
        plt.imshow(img_rgb_adTh_com_sk3,cmap='gray');plt.title("Thresholded RGB Channels Combined Logically with OR and Skeletonized K3");
        plt.show(7)
        plt.close('all')
        
        plt.figure(8)
        plt.figure(figsize=[40,40])
        plt.imshow(img_rgb_adTh_com_sk4,cmap='gray');plt.title("Thresholded RGB Channels Combined Logically with OR and Skeletonized K4");
        plt.show(8)
        plt.close('all')
    
    if s_pstage > 1:
        str_004="ps004_rgb_aBT_comb_sk_"+image_in
        cv2.imwrite(str_004, img_rgb_adTh_com_sk3)
    
    # Denoising
    img_rgb_adTh_com_sk3_bl = cv2.bilateralFilter(img_rgb_adTh_com_sk3,3,0,0);
    if plot_graphs > 0:
        plt.figure(9)
        plt.figure(figsize=[40,40])
        plt.imshow(img_rgb_adTh_com_sk3_bl,cmap='gray')
        plt.show(9)
        plt.close('all')
    
    if s_pstage > 1:
        str_005="ps005_rgb_aBT_comb_sk_dn_"+image_in
        cv2.imwrite(str_005, img_rgb_adTh_com_sk3_bl)
    else: 
        str_005="ps005_rgb_aBT_comb_sk_dn_"+image_in
        cv2.imwrite(str_005, img_rgb_adTh_com_sk3_bl)
    
    #%% Image Segmentation to Create Image with Bounding Boxes Around Text
    #   (For Visualization. Actual segmentation is done in Tesseract)
    
    img_rgb_adTh_com_sk3_bl_boxes = img_rgb_adTh_com_sk3_bl
    d = pytesseract.image_to_data(img_rgb_adTh_com_sk3_bl_boxes, output_type=Output.DICT)
    # print(d.keys()) #uncomment to print out other items in 'd.keys'
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img_rgb_adTh_com_sk3_bl_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

    str_006="ps006_rgb_aBT_comb_sk_dn_boxes_"+image_in
    cv2.imwrite(str_006, img_rgb_adTh_com_sk3_bl_boxes)
    final_im = img_rgb_adTh_com_sk3_bl_boxes;
    
    #%% Extract Text from the Image with Tesseract OCR
    
    #build UNIX command string
    img_noExt = os.path.splitext(image_in)[0]
    txt_out_file = img_noExt+out_file_tag
    ts_command ="tesseract "+str_005+" "+txt_out_file
    
    # Execute tesseract BASH command
    os.system(ts_command)
    
    # Grab text output file
    text_out = txt_out_file+".txt"
    text_out = open(text_out,'r')
    # Return to image directory
    
    #%% Wrap Up
    if p_pstage == 1:
        ind_ps = [img, img_r, img_g, img_b, img_r_adTh, img_g_adTh, img_b_adTh, img_rgb_adTh_com, img_rgb_adTh_com_sk3, img_rgb_adTh_com_sk3_bl, img_rgb_adTh_com_sk3_bl_boxes];
    else:
        ind_ps = [];

    return text_out, final_im, ind_ps

#%% ts_all

def ts_all(img_dir,tag, out_dir, s_pstage, p_pstage):
    # img_dir : the directory with your images
    # tag = string tag which singles out all the files you want. 
    #       NOTE: Flank with with astrixes ("*tag*")
    #
    # ts_pipeline(image_dir, image_in, out_dir, s_pstage, p_pstage)
    #%% Import Packages
    import glob
    import os
    import img2txt
    #%% Get list of the desired images
    
    # enter the directory containing images
    os.chdir(img_dir)
    
    # get list of images containing tag
    file_list = glob.glob(tag)
    n_files = len(file_list)
    #%% Loop through all the images feeding each through the ts_pipeline
    for i in range(n_files-1):
        img2txt.ts_pipeline(img_dir, file_list[i], out_dir, s_pstage, p_pstage)             

