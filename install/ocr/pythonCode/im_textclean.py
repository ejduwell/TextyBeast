#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:23:14 2022

@author: eduwell
"""

def im_textclean(im_dir,out_dir,img):
    
    #%% Import Packages
    import os
    import glob
    
    #%% Code
    
    #%% Find Image
    
    # go to directory
    os.chdir(im_dir)
    
    #%% Run textcleaner with desired params..

    # base_str = os.path.splitext(img)[0]
    
    # th_im = base_str+"_loc_th.jpeg"
    
    # th_cmd = "localthresh"+" "+"-r 10"+" "+img+" "+th_im
    
    # os.system(th_cmd)
    
    base_str = os.path.splitext(img)[0]
    ext_in = os.path.splitext(img)[1]
    
    tc_cmd_str = "textcleaner"+" "+"-g -f 15 -s 1"+" "+img+" "+base_str+"_tc"+ext_in
    os.system(tc_cmd_str)

def im_textclean_rc(im_dir,out_dir,tag):
    
    #%% Import Packages
    import os
    import glob
    
    #%% Code
    
    #%% Find Images
    
    # go to directory
    os.chdir(im_dir)
    
    #Get list of images with tag in filename
    im_list = glob.glob(tag)
    
    im_list.sort() # make sure they are sorted in ascending order
    #n_im = len(im_list) # get the # of images in the list
    
    #%% Run textcleaner recursively on list of images with desired params..

    # base_str = os.path.splitext(img)[0]
    
    # th_im = base_str+"_loc_th.jpeg"
    
    # th_cmd = "localthresh"+" "+"-r 10"+" "+img+" "+th_im
    
    # os.system(th_cmd)
    
    for images in im_list:
        base_str = os.path.splitext(images)[0]
        ext_in = os.path.splitext(images)[1]
    
        tc_cmd_str = "textcleaner"+" "+"-g -f 15 -s 1"+" "+images+" "+base_str+"_tc"+ext_in
        os.system(tc_cmd_str)
  
# TEST CALL
# im_dir = ""
# out_dir = ""
# img = ""
# #img_out = ""

# im_textclean(im_dir,out_dir,img)