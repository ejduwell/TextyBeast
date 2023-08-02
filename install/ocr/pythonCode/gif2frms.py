#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:22:46 2022

@author: eduwell
"""

#%% Import Packages
from PIL import Image
from PIL import GifImagePlugin
import os
from datetime import datetime

#%% Functions
def get_frames(gif_in,gif_dir):
    
    # go to gif directory
    os.chdir(gif_dir)
    
    
    # get filename root..
    gif_in_root = os.path.splitext(gif_in)[0]
    
    # make output directory for frames
    
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    
    # create unique dir name by taking input file name and adding "_frames_"+date (format: mo-day-year-hour-min-sec) at end
    outdir_name = gif_in_root+"_frames_"+dt_string
    os.mkdir(outdir_name)
    
    
    
    # Reading an animated GIF file using Python Image Processing Library - Pillow
    imageObject = Image.open(gif_in)
    print(imageObject.is_animated)
    print(imageObject.n_frames)
    
    #enter the output directory
    os.chdir(outdir_name)
    
    # Save individual frames from the loaded animated GIF file
    itr8r = 1; # initialize iterater..
    for frame in range(0,imageObject.n_frames):
        #im_namestr = gif_in_root+"_frm_"+str(itr8r).zfill(3)+".bmp"
        im_namestr = gif_in_root+"_frm_"+str(round((itr8r/imageObject.n_frames)*1000)).zfill(3)+".bmp"
        imageObject.seek(frame)
        
        imageObject.save(im_namestr)
        itr8r = itr8r+1
    return outdir_name

#%% Test Call

# gif_in = "test2.gif"
# gif_dir = "/Users/eduwell/OneDrive - mcw.edu/Documents/SNAP/projects/Priyanka_Faces/quest_dev/faces/out"

# test_outdir = get_frames(gif_in,gif_dir)
# pause = ""









