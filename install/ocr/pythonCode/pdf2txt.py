#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:15:15 2022

http://www.fmwconcepts.com/imagemagick/textcleaner/index.php
http://www.fmwconcepts.com/imagemagick/index.php

@author: eduwell
"""
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
    text = text.replace('\n', ' ')
    
    # Finally, write the processed text to the file.
    f.write(text)
      
    # Close the file after writing all the text.
    f.close()
    
def pdf2txt(pdf_filename, pdf_dir, txt_outname, txt_outdir, scrape_imgs):
    import os
    import pdfplumber
    #import imagemagick
    os.chdir(pdf_dir)
    import sys
    import img2txt
    import im_textclean as imtc
    import time
    import frm2txt_DL
    import shutil
    from PIL import Image
    import glob
    import frm2txt_DL2
    
    with pdfplumber.open(pdf_filename) as pdf:
        pages = pdf.pages
        n_pages = len(pages)
        with open(txt_outname, 'w') as f:
            meta_data_header = "METADATA:"
            f.write(meta_data_header)
            f.write('\n')
            f.write("--------------------------------------------------------")
            f.write('\n')
            f.write('\n')
            metadata = str(pdf.metadata)
            f.write(metadata)
            f.write('\n')
            f.write("--------------------------------------------------------")
            f.write('\n')
            f.write('\n')
            for ii in range(0,n_pages):
                f.write('\n')
                f.write('\n')
                strz = "*************************************"
                slide_str = "***************PAGE_"+str(ii+1).zfill(3)+"**************"
                print(strz)
                print(slide_str)
                print(strz)
                print(" ")
                f.writelines(strz)
                f.write('\n')
                f.writelines(slide_str)
                f.write('\n')
                f.writelines(strz)
                f.write('\n')
                f.write('\n')
                f.write("-------------------------------")
                f.write('\n')
                f.write("PLAIN_TEXT_READ_FROM_THIS_PAGE:")
                f.write('\n')
                f.write("-------------------------------")
                f.write('\n')
                f.write('\n')
                page = pdf.pages[ii]
                f.write(page.extract_text())
                f.write('\n')
                f.write('\n')
                print(page.extract_text())
                print(" ")
                f.write('\n')
                f.write('\n')
                
                if scrape_imgs == 1:
                    f.write("--------------------------------------")
                    f.write('\n')
                    f.write("TEXT_SCRAPED_FROM_IMAGES_ON_THIS_PAGE:")
                    f.write('\n')
                    f.write("--------------------------------------")
                    f.write('\n')
                    f.write('\n')
                    if len(page.images) > 0:
                        for oo in range(0,len(page.images)):
                            f.write('\n')
                            im_count_Str = "Image_"+str(oo+1)+":"
                            f.write('\n')
                            f.write(im_count_Str)
                            f.write('\n')                        
                            
                            px0 = 0
                            ptop = 0
                            px1 = page.width
                            pbottom = page.height
                            
                            im = page.images[oo]
                            
                            # Check whether the image bbox falls off the page bounds..
                            # if it does, cut it off at the page boundary..
                            if im['x0'] < px0:
                                im['x0'] = px0
                                
                            if im['top'] < ptop:
                                im['top'] = ptop
                            
                            if im['x1'] > px1:
                                im['x1'] = px1
                                
                            if im['bottom'] > pbottom:
                                im['bottom'] = pbottom
                            
                            bounding_box = (im['x0'], im['top'], im['x1'], im['bottom'])
                            image = page.crop(bounding_box, relative=True)
                            image = image.to_image(resolution=150)
                            
                            image.save('temp.jpeg', format="jpeg")
    
                            #imtc.im_textclean(pdf_dir,pdf_dir,'temp.jpeg')
                            
                            print("*****DETECTING TEXT FROM IMAGE FRAMES WITH MMOCR*****")
                            begin_time = time.perf_counter()
                            # run individual frames through mmocr to extract text
                            
                            mmocr_dir = "mmocr_out/"
                            if not os.path.exists(mmocr_dir):
                                os.makedirs(mmocr_dir)               
                            
                            mmocr_dir = pdf_dir+mmocr_dir
                            detector='TextSnake';
                            recognizer='SAR';
                            
                            # Copy in the "configs" directory for mmocr if its not already there..
                            source_config = "/Users/eduwell/opt/anaconda3/lib/python3.9/configs"
                            destination_config = pdf_dir+"configs/"
                            
                            if not os.path.exists(destination_config):
                                shutil.copytree(source_config, destination_config)
                            
                            #frm2txt_DL.frm2txt_mmocr_det(pdf_dir,'temp_tc.jpeg',mmocr_dir, detector) #orig
                            
                            
                            detector='PANet_IC15'; # specifies detector to be used used by mmocr to detect/put bounding boxes around text
                            #detector='TextSnake'
                            recognizer='SAR'; # specifies the recognizer which interprets/"reads" the text image within bounding boxes
    
                            x_merge = 65; # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box
                            frm2txt_DL2.frm2txt_mmocr_det(pdf_dir,'*temp.jpeg*',"mmocr_out", detector,recognizer,x_merge)
                            
                            #frm2txt_DL.bbox_txtMask_snek(pdf_dir, mmocr_dir, "temp_tc.jpeg","out_temp_tc.json", 2) #orig
                            
                            ClustThr_factor = 3;
                            txt_out = frm2txt_DL2.bbox_txtMask_snek(mmocr_dir, mmocr_dir, "*_mmorc_PANet_IC15.jpg*","*temp.json*", ClustThr_factor, "euclidean", "ward")
                            
                            os.chdir(pdf_dir)
                            
                            frame_itr = 0;
    
                            for frames in txt_out:
                                      
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

def pdf2txt_layout(pdf_filename, pdf_dir, txt_outname, txt_outdir, scrape_imgs):
    import os
    import pdfplumber
    #import imagemagick
    os.chdir(pdf_dir)
    import sys
    import img2txt
    import im_textclean as imtc
    import time
    import frm2txt_DL
    import shutil
    from PIL import Image
    import glob
    import frm2txt_DL2
    
    with pdfplumber.open(pdf_filename) as pdf:
        pages = pdf.pages
        n_pages = len(pages)
        with open(txt_outname, 'w') as f:
            meta_data_header = "METADATA:"
            f.write(meta_data_header)
            f.write('\n')
            f.write("--------------------------------------------------------")
            f.write('\n')
            f.write('\n')
            metadata = str(pdf.metadata)
            f.write(metadata)
            f.write('\n')
            f.write("--------------------------------------------------------")
            f.write('\n')
            f.write('\n')
            for ii in range(0,n_pages):
                f.write('\n')
                f.write('\n')
                strz = "*************************************"
                slide_str = "***************PAGE_"+str(ii+1).zfill(3)+"**************"
                print(strz)
                print(slide_str)
                print(strz)
                print(" ")
                f.writelines(strz)
                f.write('\n')
                f.writelines(slide_str)
                f.write('\n')
                f.writelines(strz)
                f.write('\n')
                f.write('\n')
                f.write("-------------------------------")
                f.write('\n')
                f.write("PLAIN_TEXT_READ_FROM_THIS_PAGE:")
                f.write('\n')
                f.write("-------------------------------")
                f.write('\n')
                f.write('\n')
                page = pdf.pages[ii]
                f.write(page.extract_text(layout=True))
                f.write('\n')
                f.write('\n')
                print(page.extract_text(layout=True))
                print(" ")
                f.write('\n')
                f.write('\n')
                
                if scrape_imgs == 1:
                    f.write("--------------------------------------")
                    f.write('\n')
                    f.write("TEXT_SCRAPED_FROM_IMAGES_ON_THIS_PAGE:")
                    f.write('\n')
                    f.write("--------------------------------------")
                    f.write('\n')
                    f.write('\n')
                    if len(page.images) > 0:
                        for oo in range(0,len(page.images)):
                            f.write('\n')
                            im_count_Str = "Image_"+str(oo+1)+":"
                            f.write('\n')
                            f.write(im_count_Str)
                            f.write('\n')                        
                            
                            px0 = 0
                            ptop = 0
                            px1 = page.width
                            pbottom = page.height
                            
                            im = page.images[oo]
                            
                            # Check whether the image bbox falls off the page bounds..
                            # if it does, cut it off at the page boundary..
                            if im['x0'] < px0:
                                im['x0'] = px0
                                
                            if im['top'] < ptop:
                                im['top'] = ptop
                            
                            if im['x1'] > px1:
                                im['x1'] = px1
                                
                            if im['bottom'] > pbottom:
                                im['bottom'] = pbottom
                            
                            bounding_box = (im['x0'], im['top'], im['x1'], im['bottom'])
                            image = page.crop(bounding_box, relative=True)
                            image = image.to_image(resolution=150)
                            
                            image.save('temp.jpeg', format="jpeg")
    
                            #imtc.im_textclean(pdf_dir,pdf_dir,'temp.jpeg')
                            
                            print("*****DETECTING TEXT FROM IMAGE FRAMES WITH MMOCR*****")
                            begin_time = time.perf_counter()
                            # run individual frames through mmocr to extract text
                            
                            mmocr_dir = "mmocr_out/"
                            if not os.path.exists(mmocr_dir):
                                os.makedirs(mmocr_dir)               
                            
                            mmocr_dir = pdf_dir+mmocr_dir
                            detector='TextSnake';
                            recognizer='SAR';
                            
                            # Copy in the "configs" directory for mmocr if its not already there..
                            source_config = "/Users/eduwell/opt/anaconda3/lib/python3.9/configs"
                            destination_config = pdf_dir+"configs/"
                            
                            if not os.path.exists(destination_config):
                                shutil.copytree(source_config, destination_config)
                            
                            #frm2txt_DL.frm2txt_mmocr_det(pdf_dir,'temp_tc.jpeg',mmocr_dir, detector) #orig
                            
                            
                            detector='PANet_IC15'; # specifies detector to be used used by mmocr to detect/put bounding boxes around text
                            #detector='TextSnake'
                            recognizer='SAR'; # specifies the recognizer which interprets/"reads" the text image within bounding boxes
    
                            x_merge = 65; # Controls how far laterally (in the x direction) mmocr will look to merge text boxes into same box
                            frm2txt_DL2.frm2txt_mmocr_det(pdf_dir,'*temp.jpeg*',"mmocr_out", detector,recognizer,x_merge)
                            
                            #frm2txt_DL.bbox_txtMask_snek(pdf_dir, mmocr_dir, "temp_tc.jpeg","out_temp_tc.json", 2) #orig
                            
                            ClustThr_factor = 3;
                            txt_out = frm2txt_DL2.bbox_txtMask_snek(mmocr_dir, mmocr_dir, "*_mmorc_PANet_IC15.jpg*","*temp.json*", ClustThr_factor, "euclidean", "ward")
                            
                            os.chdir(pdf_dir)
                            
                            frame_itr = 0;
    
                            for frames in txt_out:
                                      
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
        
def pdf2txt_tag(tag, pdf_dir, txt_outdir,scrape_imgs):
    import glob
    import os
    
    os.chdir(pdf_dir)


    file_list = glob.glob(tag)
    file_list.sort() # make sure they are sorted in ascending order
    #n_files = len(file_list)
    
    for files in file_list:
        txt_outname = os.path.splitext(files)[0]+".txt"
        pdf2txt(files, pdf_dir, txt_outname, txt_outdir, scrape_imgs)

def pdf2txt_layout_tag(tag, pdf_dir, txt_outdir,scrape_imgs):
    import glob
    import os
    
    os.chdir(pdf_dir)


    file_list = glob.glob(tag)
    file_list.sort() # make sure they are sorted in ascending order
    #n_files = len(file_list)
    
    for files in file_list:
        txt_outname = os.path.splitext(files)[0]+".txt"
        pdf2txt_layout(files, pdf_dir, txt_outname, txt_outdir, scrape_imgs)

########################## test calls ##########################
################################################################
# in_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/"
# out_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/"
# scrape_imgs = 0;

# pdf2txt_layout_tag("*.pdf*", in_dir, out_dir, scrape_imgs)
