#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:19:13 2022

tutorals:
    https://colab.research.google.com/github/open-mmlab/mmocr/blob/main/demo/MMOCR_Tutorial.ipynb#scrollTo=iQQIVH9ApEUv
    https://colab.research.google.com/github/open-mmlab/mmocr/blob/main/demo/MMOCR_Tutorial.ipynb#scrollTo=oALHgzmrAqik
    
    NOTE: FIGURE THIS OUT: It is recommended to symlink the dataset root to mmocr/data. 
          Please refer to datasets.md to prepare your datasets. If your folder structure is different, 
          you may need to change the corresponding paths in config files.

          ? https://askubuntu.com/questions/986054/how-to-create-a-symlink-to-root
          
Look into:
    https://mmocr.readthedocs.io/en/latest/textdet_models.html
    https://mmocr.readthedocs.io/en/latest/textrecog_models.html
    https://mmocr.readthedocs.io/en/latest/tutorials/blank_recog.html
    
    
    EJD Notes:
        In order to make the installation of mmocr work, I needed to install the mmocr directory in the
        site-packages directory of anaconda. **I also needed to compile the package mmcv-full from source
        myself.. there were secret voodoo directions here:
            https://mmcv.readthedocs.io/en/latest/get_started/build.html
@author: eduwell
"""


def rctgle_chk(bl, tr, p):  # bl = bottom left (x,y), tr = top right (x,y), p = point coordinate (x,y)
    if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]):
        return True
    else:
        return False
#bottom_left = (1, 1)
#top_right = (8, 5)
#point = (5, 4)
#print(rctgle_chk(bottom_left, top_right, point))


#import frm2txt_DL
#frm2txt_DL.frm2txt_mmocr('/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/MMOCR_TEST/test_in_dir','*frame*', 'out_dir', 'PANet_IC15', 'SAR')
def contourIntersect(original_image, contour1, contour2):
    import cv2
    import numpy as np

    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()


def frm2txt_mmocr(img_dir, tag, out_dir, detector, recognizer):

    # %% Import Modules

    #import importlib.util
    #spec = importlib.util.spec_from_file_location("mmocr", "/Users/eduwell/mmocr/mmocr/__init__.py")
    #mmocr = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mmocr)
    # mmocr.MyClass()
    import os
    #work_dir = os.getcwd()
    # os.chdir('/Users/eduwell/mmocr')
    import sys
    import cv2
    #import mmocr
    from mmocr.utils.ocr import MMOCR
    # sys.path.append('/Users/eduwell/mmocr')
    # sys.path.append('/opt/homebrew/lib/python3.9/site-packages/mmcv/utils')
    #import matplotlib.pyplot as plt
    import os
    import glob
    import os
    import numpy as np
    #import img2txt

    # %% Parameters
    out_dir = img_dir+'/'+out_dir
    mmocr = MMOCR(det=detector, recog=recognizer)
    #mmocr = mmocr.ocr(det=detector, recog=recognizer)
    #mmocr = MMOCR(det='PANet_IC15', recog='SAR')
    # %% Get the images
    # enter the directory containing images
    strt_dir = os.getcwd()
    os.chdir(img_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get list of images containing tag
    file_list = glob.glob(tag)
    n_files = len(file_list)
    os.chdir(strt_dir)

    # Loop through all the images feeding each through the ts_pipeline
    # get list of images containing tag
    #file_list = glob.glob(tag)
    #n_files = len(file_list)
    for i in range(n_files):
        img_in = img_dir+'/'+file_list[i]
        img_out = out_dir+'/'+file_list[i] + \
            '_mmorc_'+detector+'_'+recognizer+'.jpg'
        #mmocr.readtext(img_in, print_result=True, output=img_out)
        results = mmocr.readtext(img_in, print_result=True, output=img_out,
                                 merge=True, merge_xdist=60, export=out_dir, details=True)


def frm2txt_mmocr_det(img_dir, tag, out_dir, detector,recognizer,x_merge,det_ckpt_in,recog_ckpt_in):

    # %% Import Modules

    #import importlib.util
    #spec = importlib.util.spec_from_file_location("mmocr", "/Users/eduwell/mmocr/mmocr/__init__.py")
    #mmocr = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mmocr)
    # mmocr.MyClass()
    import os
    #work_dir = os.getcwd()
    # os.chdir('/Users/eduwell/mmocr')
    import sys
    import cv2
    #import mmocr
    #from mmocr.utils.ocr import MMOCR
    from mmocr.apis import MMOCRInferencer
    
    # sys.path.append('/Users/eduwell/mmocr')
    # sys.path.append('/opt/homebrew/lib/python3.9/site-packages/mmcv/utils')
    #import matplotlib.pyplot as plt
    import os
    import glob
    import os
    import numpy as np
    import torch
    import shutil
    from PIL import Image
    #import img2txt

    # %% Parameters
    out_dir = img_dir+'/'+out_dir
    #mmocr = MMOCR(det=detector, recog=recognizer)
    
    #mmocr = MMOCR(det=detector, recog=recognizer,det_ckpt=det_ckpt_in,recog_ckpt=recog_ckpt_in) #new..(ejd 3/6/23)
    #mmocr = MMOCRInferencer(det=detector, rec=recognizer,det_weights=det_ckpt_in,rec_weights=recog_ckpt_in) #new..(ejd 6/22/23)
    #mmocr = MMOCRInferencer(det=detector, rec=recognizer,det_weights=det_ckpt_in,rec_weights=recog_ckpt_in) #new..(ejd 6/23/23)
    mmocr = MMOCRInferencer(det='DBNet', rec='SAR',device='gpu');
    
    #mmocr = mmocr.ocr(det=detector, recog=recognizer)
    #mmocr = MMOCR(det='PANet_IC15', recog='SAR')
    # %% Get the images
    # enter the directory containing images
    strt_dir = os.getcwd()
    os.chdir(img_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get list of images containing tag
    file_list = glob.glob(tag)
    n_files = len(file_list)
    os.chdir(strt_dir)

    # Loop through all the images feeding each through the ts_pipeline
    # get list of images containing tag
    #file_list = glob.glob(tag)
    #n_files = len(file_list)
    for i in range(n_files):
        img_in = img_dir+'/'+file_list[i]
        im_in_base = os.path.splitext(file_list[i])[0]
        #img_out = out_dir+'/'+im_in_base+'_mmorc_'+detector+'.jpg'
        
        #img_out = out_dir+im_in_base+'_mmorc_'+detector+".png"
        img_out = out_dir
        
        #mmocr.readtext(img_in, print_result=True, output=img_out)
        #results = mmocr.readtext(img_in, print_result=True, output=img_out, merge=True, merge_xdist=60, export=out_dir, details=True)
        #results = mmocr.readtext(img_in, print_result=True, output=img_out,
                                 #export=out_dir, details=True, merge=True, merge_xdist=x_merge)
        # empty the gpu cache
        torch.cuda.empty_cache()
        
        #EJD DEBUG:
        #=======================
        print("img_in name:")
        print(img_in)
        print("x_merge value:")
        print(x_merge)
        
        try:
            #results = mmocr.readtext(img_in, print_result=True,output=img_out,export=out_dir, details=True, merge=True, merge_xdist=x_merge)
            results = mmocr(img_in, out_dir=out_dir, save_pred=True, save_vis=True)
        except:
            print("Warning, error occured when running: results = mmocr.readtext(img_in, print_result=True,output=img_out,export=out_dir, details=True, merge=True, merge_xdist=x_merge)")
            print("E.J. Duwell put this try/catch fail-safe in place to deal with occasional errors caused by mmocr creating text boxes with dimesions equal to zero...")
            print("This had been crashing the pipeline occasionally as it created a 'divide by zero' error.. If you are seeing this message in the error log, that is likely the cause...")
            print("However, this fail-safe try/catch could also be tripped by any failure of the mmocr.readtext call above..")
            print("This occurred on image:")
            print(img_in)
            print("creating dummy .json and then continuing to next image without extracting text..")
            
            # Extract the base name without extension
            file_nameTmp = os.path.splitext(os.path.basename(img_in))[0]
            file_nameTmp="out_"+file_nameTmp
            
            # read in a copy of the image and just save copy as expected mmocr
            # output .png file (without doing anything..)
            # Open the JPEG file
            jpeg_file = img_in;
            with Image.open(jpeg_file) as img:
                # Convert the image to RGBA format
                rgba_img = img.convert('RGBA')
                # Output PNG file path
                png_file = out_dir+"/"+file_nameTmp+".png"
                # Save the image as PNG
                rgba_img.save(png_file, 'PNG')
            
            # Print out a dummy .json file with expected name and make results empty...
            dummyFileTxt="{"+"\n"+"    "+'"'+"filename"+'"'+": "+'"'+file_nameTmp+'"'+","+"\n"+"    "+'"'+"result"+'"'+": []"+"\n"+"}"
            print("DUMMY .JSON TEXT:")
            print(dummyFileTxt)
            print("")
            
            file_pathTmp = out_dir+"/"+file_nameTmp+".json"
            with open(file_pathTmp, 'w') as file:
                # Write the string to the file
                string_to_write = dummyFileTxt
                file.write(string_to_write)
            continue
        #=======================
        
        # empty the gpu cache
        torch.cuda.empty_cache()
    # empty the gpu cache
    torch.cuda.empty_cache()

def bbox_txtMask(im_dir, txt_dir, im_tag, txt_tag):
    import cv2
    import os
    import glob
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    os.chdir(im_dir)
    # get list of images containing tag
    im_list = glob.glob(im_tag)
    im_list.sort()  # make sure they are sorted in ascending order
    n_im = len(im_list)

    os.chdir(txt_dir)
    # get list of images containing tag
    txt_list = glob.glob(txt_tag)
    txt_list.sort()  # make sure they are sorted in ascending order
    n_txt = len(txt_list)

    os.chdir(im_dir)

    # Get image dimensions
    img4dims = cv2.imread(im_list[0])
    ydim, xdim, zdim = img4dims.shape

    for ii in range(0, n_im):
        img = cv2.imread(im_list[ii])
        mask = np.zeros((ydim, xdim, zdim), np.uint8)

        j_file = txt_list[ii]
        # Opening JSON file
        j_file = open(txt_dir+j_file)

        # returns JSON object as
        # a dictionary
        j_data = json.load(j_file)

        # Iterating through the json
        # list
        size_obj = len(j_data['result'])
        for i in range(0, size_obj):
            box_coord = j_data['result'][i]['box']

            # inflate the box sizes by 10% to make sure you get everything..
            #  cv2.rectangle(mask,(TL_Y, TL_X),(BR_Y,BR_X),(1,1,1),-1)

            factor = 0.15

            height = abs(box_coord[2]-box_coord[6])
            h_expd = height*(1+factor)
            h_add = round((abs(height - h_expd)/2))

            width = abs(box_coord[3]-box_coord[7])
            w_expd = width*(1+factor)
            w_add = round((abs(width - w_expd)/2))

            # draw box as rectangle in mask image
            cv2.rectangle(mask, (box_coord[2]+h_add, box_coord[3]-w_add),
                          (box_coord[6]-h_add, box_coord[7]+w_add), (1, 1, 1), -1)
            #plt.imshow(mask, cmap='plasma')

        #plt.imshow(mask, cmap='plasma')
        # plt.show()
        mask_inv = cv2.bitwise_not(mask)
        mask_str_base = os.path.splitext(im_list[ii])[0]
        mask_str = mask_str_base+"_txt_mask.jpg"
        mask_apld_str = mask_str_base+"_txt_mask_apld.jpg"
        cv2.imwrite(mask_str, mask)
        masked_im = (img*mask)+mask_inv
        cv2.imwrite(mask_apld_str, masked_im)


def bbox_txtMask_snek(im_dir, txt_dir, im_tag, txt_tag, ClustThr_factor, affinity, linkage):  # ****
    import cv2
    import os
    import glob
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    import pandas as pd
    import numpy as np
    #from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as sch
    from operator import itemgetter
    import copy
    from sklearn.cluster import OPTICS, cluster_optics_dbscan
    import matplotlib.gridspec as gridspec
    #import pytesseract #ejd commented 3/6/23
    #from pytesseract import Output  #ejd commented 3/6/23

    os.chdir(im_dir)
    # get list of images containing tag
    im_list = glob.glob(im_tag)
    im_list.sort()  # make sure they are sorted in ascending order
    n_im = len(im_list)

    os.chdir(txt_dir)
    # get list of text files containing tag
    txt_list = glob.glob(txt_tag)
    txt_list.sort()  # make sure they are sorted in ascending order
    n_txt = len(txt_list)

    os.chdir(im_dir)

    # Get image dimensions
    img4dims = cv2.imread(im_list[0], 0)
    ydim, xdim = img4dims.shape
    
    txt_out_final = []
    for ii in range(0, n_im):
        
        #EJD placed these for debugging purposes.. commented now...
        #print("Frame:")
        #print(str(ii))
        #print(str(im_list[ii]))
        
        img = cv2.imread(im_list[ii], 0)
        #mask = np.zeros((ydim,xdim,zdim), np.uint8)
        mask = np.zeros((ydim, xdim), dtype=np.uint8)

        j_file = txt_list[ii]
        # Opening JSON file
        j_file = open(txt_dir+j_file)

        # returns JSON object as
        # a dictionary
        j_data = json.load(j_file)

        # Iterating through the json
        # list
        #size_obj = len(j_data['box:'])
        
        size_obj = len(j_data['result'])
        
        
        contour_list = []
        centroid_lst = []
        centroid_lst_wh = []
        clust_heights = []
        text_list = []
        for i in range(0, size_obj):
            shape_coord = j_data['result'][i]['box']
            shape_text = j_data['result'][i]['text'] #***
            shape = []
            text_list.append(shape_text)
            xpt = int(0)
            ypt = int(1)
            npairs = int((len(shape_coord))/2)
            contour_y =[]
            contour_x =[]
            for uu in range(0, npairs):
                xpoint = int(round(shape_coord[xpt]))
                ypoint = round(int(shape_coord[ypt]))
                point = [xpoint, ypoint]
                shape.append(point)
                contour_x.append(xpoint)
                contour_y.append(ypoint)
                xpt = xpt+2
                ypt = ypt+2

            contours = np.array(shape)
            contour_xMin = min(contour_x)
            
            
            x_shp, y_shp, w_shp, h_shp = cv2.boundingRect(contours)
            M = cv2.moments(contours)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            #EJD COMMENTED BELOW TO TRY USING Xmin (far left instead of the middle to cluster..)
            #centroid = [cX, cY]
            centroid = [contour_xMin, cY]
            
            centroid_wh = [cX, cY, h_shp]
            clust_heights.append(h_shp)

            centroid_lst.append(centroid)
            centroid_lst_wh.append(centroid_wh)
            contour_list.append(contours) ###
            mask = cv2.fillPoly(mask, np.int32([contours]), 1)

        # mask_inv = np.logical_not(mask)
        # mask_inv = np.multiply(mask_inv, 1)
        # mask_inv = mask_inv*255
        # masked_im = (img*mask)+mask_inv
        # masked_im2 = masked_im.astype(np.uint8)
        # masked_im3 = copy.deepcopy(masked_im2)
        
        # masked_im4 = copy.deepcopy(masked_im2)
        # masked_im5 = copy.deepcopy(masked_im2)
        # masked_im6 = copy.deepcopy(masked_im2)

        # if d['level'][i] == 3:
        #    cv2.rectangle(masked_im5, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # if d['level'][i] == 4:
        #    cv2.rectangle(masked_im6, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #pause = "pause"
        # cv2.imwrite("test_L1.jpeg",masked_im3)
        # cv2.imwrite("test_L5.jpeg",masked_im4)

        # UNCOMMENT TO VIEW THE L2 BOXES...
        # cv2.imwrite("L2_boxes.jpeg",masked_im2)

        # cv2.imwrite("test_L3.jpeg",masked_im5)
        # cv2.imwrite("test_L4.jpeg",masked_im6)
        # factor = 0.15

        # height = abs(box_coord[2]-box_coord[6])
        # h_expd = height*(1+factor)
        # h_add = round((abs(height - h_expd)/2))

        # width = abs(box_coord[3]-box_coord[7])
        # w_expd = width*(1+factor)
        # w_add = round((abs(width - w_expd)/2))

        # #draw box as rectangle in mask image
        # cv2.rectangle(mask,(box_coord[2]+h_add,box_coord[3]-w_add),(box_coord[6]-h_add,box_coord[7]+w_add),(1,1,1),-1)
        # #plt.imshow(mask, cmap='plasma')

        #plt.imshow(mask, cmap='plasma')
        # plt.show()
        #mask_inv = cv2.bitwise_not(mask)

        # make dendrogram of clustering for visualization

        if len(contour_list) > 1: #ORIG
        #if len(contour_list) >= 1: #EJD EDIT #***
        #if len(contour_list) > -1: #EJD EDIT #***

            #dendrogram = sch.dendrogram(sch.linkage(centroid_lst,affinity='euclidean', method='single'))

            # Fit the clustering model and get labels for centroid list
            #d_thr = 0.1*(ydim)

            # EJD ALTERNATE CLUSTERING METHOD ATTEMPTS:
            # model = OPTICS(min_samples=1, xi=0.05, min_cluster_size=0.05)

            # model.fit(centroid_lst)
            # labels = model.labels_

            # from sklearn.cluster import DBSCAN
            # model = DBSCAN(eps=0.3, min_samples=2).fit(centroid_lst)
            # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
            # core_samples_mask[model.core_sample_indices_] = True
            # labels = model.labels_

            # ORIG CLUSTERING MODEL!!
            # ORIG CLUSTERING MODEL!!

            # d_thr = 2*(np.mean(clust_heights)) # commented

            d_thr = ClustThr_factor*(np.mean(clust_heights))

            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=d_thr, affinity=affinity, linkage=linkage)

            model.fit(centroid_lst)
            # model.fit(centroid_lst_wh)
            labels = model.labels_
            # ORIG CLUSTERING MODEL!!
            # ORIG CLUSTERING MODEL!!

            labels_lst = labels.tolist()
            for nn in range(0, len(labels_lst)):
                labels_lst[nn] = str(labels_lst[nn])

            # Get number of unique clusters
            a_list = labels_lst
            a_set = set(a_list)
            n_uVals = len(a_set)
            
            
            
                
            #clLabeled_contours = np.vstack((labels_lst, contour_list))
            
            clLabeled_contours = np.zeros((3,len(contour_list)), object)
            
            for bb in range(0,len(labels_lst)):
                clLabeled_contours[0][bb] = labels_lst[bb]
                
            for bb in range(0,len(contour_list)):
                clLabeled_contours[1][bb] = contour_list[bb]
                
            for bb in range(0,len(contour_list)):
                clLabeled_contours[2][bb] = text_list[bb]

            #lbl_cntrs_cntroids = np.vstack((labels_lst,contour_list,centroid_lst))
            # lbl_cntrs_cntroids.sort(0)
            
            clusters_txt = []
            clusters = []
            
            for yy in a_set:
                c_string = "cluster_"+yy
                cmd_str1 = c_string+"=[]"
                exec(cmd_str1)
                cmd_str2 = "clusters.append("+c_string+")"
                exec(cmd_str2)
                del yy
                
            for yy in a_set:
                c_string = "cluster_"+yy+"_txt"
                cmd_str1 = c_string+"=[]"
                exec(cmd_str1)
                cmd_str2 = "clusters_txt.append("+c_string+")"
                exec(cmd_str2)
                del yy

            cluster_centroids = copy.deepcopy(clusters)

            for bb in range(0, (clLabeled_contours.shape[1])):
                cl_idx = clLabeled_contours[0, bb]
                clusters[int(cl_idx)].append(clLabeled_contours[1, bb])
                cluster_centroids[int(cl_idx)].append(centroid_lst[bb])
            del bb
            
            
            
            
            for bb in range(0, (clLabeled_contours.shape[1])):
                cl_idx = clLabeled_contours[0, bb]
                clusters_txt[int(cl_idx)].append(clLabeled_contours[2, bb])
            del bb

            # Get the average centroid coordinate for text boxes in each cluster (ie get the cluster centroid)
            ave_cluster_centroid_x = []
            ave_cluster_centroid_y = []
            ave_cluster_centroids = []
            cl_xyMaxMin = []

            fstpass = 1
            # for bb in cluster_centroids:
            for bb in clusters:
                xlist = []
                ylist = []
                for yy in bb:
                    for vv in yy:
                        xvalue = vv[0]
                        xlist.append(xvalue)
                        yvalue = vv[1]
                        ylist.append(yvalue)
                x_ave = np.mean(xlist)
                y_ave = np.mean(ylist)

                cx_max = max(xlist)
                cy_max = max(ylist)
                cx_min = min(xlist)
                cy_min = min(ylist)

                mean_cntrd = [x_ave, y_ave]
                ave_cluster_centroid_x.append(x_ave)
                ave_cluster_centroid_y.append(y_ave)
                ave_cluster_centroids.append(mean_cntrd)
                if fstpass == 1:
                    cl_xyMaxMin = [cx_max, cx_min, cy_max, cy_min]
                    if len(clusters) == 1:
                        # Convert the list to a one-row NumPy array
                        cl_xyMaxMin = np.array(cl_xyMaxMin, ndmin=2)
                        
                if fstpass == 0:
                    cl_xyMaxMin = np.vstack(
                        (cl_xyMaxMin, [cx_max, cx_min, cy_max, cy_min]))
                fstpass = 0
                # cl_xyMaxMin.append([cx_max,cx_min,cy_max,cy_min])
            del bb
            del yy

            #cl_labels_cntrds = list(a_set)
            cl_labels_cntrds = list(range(0, n_uVals))

            cluster_mat = np.vstack((copy.deepcopy(cl_labels_cntrds), copy.deepcopy(ave_cluster_centroid_x)))
            cluster_mat = np.vstack((cluster_mat, copy.deepcopy(ave_cluster_centroid_y)))
            cl_xyMaxMin = np.transpose(cl_xyMaxMin)
            
            #EJD NOTE : THIS IS WHERE ERROR IS OCCURING IN SELECT VIDEOS: File "/scratch/g/tark/dataScraping/envs/ocr/env/lib/python3.9/site-packages/frm2txt_DL2.py", line 566, in bbox_txtMask_snek cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin)) 
            # File "<__array_function__ internals>", line 200, in vstack
            # File "/scratch/g/tark/dataScraping/envs/ocr/env/lib/python3.9/site-packages/numpy/core/shape_base.py", line 296, in vstack
            # return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
            # File "<__array_function__ internals>", line 200, in concatenate
            # ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1 and the array at index 1 has size 4
            
            cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin))
            
            #BELOW WAS FOR DEBUGGING PURPOSES (to sort out error above..) NOW COMMENTED..
            #try:
            #    cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin))
            #    print("okokokokokokokokokokokokokokokokokokokokokokokokokokokok")
            #    print("CLUSTER_MAT:")
            #    print(cluster_mat)
            #    print("")
            #    print("cl_xyMaxMin")
            #    print(cl_xyMaxMin)
            #    print("")
            #    print("len(clusters):")
            #    print(len(clusters))
            #    print("okokokokokokokokokokokokokokokokokokokokokokokokokokokok")
            #    print("")
            #    cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin))
            #except:
            #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #    print("CLUSTER_MAT:")
            #    print(cluster_mat)
            #    print("")
            #    print("cl_xyMaxMin")
            #    print(cl_xyMaxMin)
            #    print("")
            #    print("len(clusters):")
            #    print(len(clusters))
            #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #    print("")
            #    cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin))
                
            ave_cluster_centroid_y_sort = np.vstack((copy.deepcopy(cl_labels_cntrds), copy.deepcopy(ave_cluster_centroid_y)))
            ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:,ave_cluster_centroid_y_sort[1, :].argsort()]
            #cluster_centroids_labeled = np.vstack((cl_labels_cntrds,ave_cluster_centroid_x,ave_cluster_centroid_y))

            #cluster_centroids_labeled = cluster_centroids_labeled[:, cluster_centroids_labeled[2, :].argsort()]
            #cluster_centroids_labeled = cluster_centroids_labeled[:, cluster_centroids_labeled[2, :].argsort(kind='mergesort')]
            #pause = "pause"

            cluster_labels_sorted = ave_cluster_centroid_y_sort[0]
            
            # EJD: NOTE: ETHAN "THOUGHT OUTSIDE THE BOX"
            # NEED TO REWORK DESCRIPTION BELOW!!
            #
            # ORDER OF SERVICE FOR SLIDE SEGMENTATION:
            # 1) Find Heading bbox
            # 2) Cut page @ ypoint to cut off header
            # 3) Get list of all possible vertical cutpoints below header
            # 4) In space below the header iterate through starting at 0
            # cuts looking for min # of cuts to get segments with only 1
            # text cluster per "row" in the image

            # Go through the bounding boxes/snakeshapes in "contour_list" and
            # segment the frame into chunks by finding blank vertical corridors
            # without text boxes spaces and bisecting them with vertical lines

            xy_MaxMin = []
            bbox_siz = []
            ymid_pos = []
            for b_boxes in contour_list:
                # for each b_box contour, find the maximum and minimum x and
                # y valuse
                x_vls = []
                y_vls = []
                for points in b_boxes:
                    y_vls.append(points[1])
                    x_vls.append(points[0])

                xmax = max(x_vls)
                xmin = min(x_vls)
                ymax = max(y_vls)
                ymin = min(y_vls)

                xy_MaxMin.append([xmax, xmin, ymax, ymin])
                wid = xmax - xmin
                hgt = ymax - ymin
                ymid = ((ymax + ymin)/2)
                siz = (wid*hgt)
                bbox_siz.append(siz)
                ymid_pos.append(ymid)

            del b_boxes

            # (1) FIND THE HEADER BBOX: Look for the largest bbox in the top 1/Nth % of the page
            size_vY = np.vstack((bbox_siz, ymid_pos))

            # grab the bboxes in size_vY in top 1/nth of pg..
            # NOTE 0,0 is at the top left corner of page...
            nthr = 0.18
            y_topNth = ((ydim)*(nthr))

            size_vY_topNth = np.zeros(
                (size_vY.shape[0], size_vY.shape[1]), dtype=np.uint8)

            for yy in range(0, (size_vY.shape[1]-1)):
                if size_vY[1][yy] <= y_topNth:
                    size_vY_topNth[0][yy] = size_vY[0][yy]
                    size_vY_topNth[1][yy] = size_vY[1][yy]

            if max(size_vY_topNth[0][:]) > 0:

                header_ymin = y_topNth
                # EJD COMMENTED THE BELOW LINES TO TRY SOMETHING SIMPLER.. (Just CUT at threshold if there's something in the region)
                # header_idx = np.argmax(size_vY_topNth[0][:]) # Get the index biggest box in the top Nth...
                #header_ymin = xy_MaxMin[header_idx][2]

            else:
                header_ymin = 0

            # 2) Cut page @ ypoint to cut off header
            # 3) Get list of all possible vertical cutpoints below header

            # preallocate zero-filled vectors for the x and y frame dimensions
            # which are the length of the x and y dimensions respectively

            xdim_binTxt = np.zeros((1, xdim), dtype=np.uint8)
            ydim_binTxt = np.zeros((1, ydim), dtype=np.uint8)

            for b_boxes in xy_MaxMin:
                xrng = list(range(b_boxes[1], b_boxes[0]))
                yrng = list(range(b_boxes[3], b_boxes[2]))

                if b_boxes[3] > header_ymin:
                    for nmbrs in xrng:
                        xdim_binTxt[0][nmbrs] = 1

                #del nmbrs
                if b_boxes[3] > header_ymin:
                    #for nmbrs in yrng:
                    for nmbrs in range(b_boxes[3], b_boxes[2]-1): #EJD ATTEMPT TO FIX ERROR BELOW #***
                        maxVal=len(ydim_binTxt[0][:])-1
                        #print("maxVal:")
                        #print(str(maxVal))
                        if nmbrs>maxVal:
                            nmbrs=maxVal
                        ydim_binTxt[0][nmbrs] = 1 #!! FIX: IndexError: index 1080 is out of bounds for axis 0 with size 1080

            xdim_gaps_b = []
            xdim_gaps_e = []
            ydim_gaps_b = []
            ydim_gaps_e = []

            for ww in range(0, xdim_binTxt.shape[1]):

                if xdim_binTxt[0][ww] == 0 and ww == 0:
                    xdim_gaps_b.append(ww)

                if xdim_binTxt[0][ww] == 0 and xdim_binTxt[0][ww-1] == 1:
                    xdim_gaps_b.append(ww)

                if ww != (xdim_binTxt.shape[1]-1):
                    if xdim_binTxt[0][ww] == 0 and xdim_binTxt[0][ww+1] == 1:
                        xdim_gaps_e.append(ww)

                if xdim_binTxt[0][ww] == 0 and ww == (xdim_binTxt.shape[1]-1):
                    xdim_gaps_e.append(ww)

            del ww

            for ww in range(0, ydim_binTxt.shape[1]):

                if ydim_binTxt[0][ww] == 0 and ww == 0:
                    ydim_gaps_b.append(ww)

                if ydim_binTxt[0][ww] == 0 and ydim_binTxt[0][ww-1] == 1:
                    ydim_gaps_b.append(ww)

                if ww != (ydim_binTxt.shape[1]-1):
                    if ydim_binTxt[0][ww] == 0 and ydim_binTxt[0][ww+1] == 1:
                        ydim_gaps_e.append(ww)

                if ydim_binTxt[0][ww] == 0 and ww == (ydim_binTxt.shape[1]-1):
                    ydim_gaps_e.append(ww)

            del ww

            # take the mean of the beginning and end indicies for each gap to
            # get cutpoints for x and y dimensions..

            x_cutpnts = []
            x_cutpnts_siz = []
            y_cutpnts = []
            y_cutpnts_siz = []

            itr8r = 0
            for ww in xdim_gaps_b:
                xcut = ((xdim_gaps_b[itr8r] + xdim_gaps_e[itr8r])/2)
                xsiz = (xdim_gaps_e[itr8r]-xdim_gaps_b[itr8r])
                x_cutpnts.append(xcut)
                x_cutpnts_siz.append(xsiz)
                itr8r = itr8r+1
            x_cutpnts = np.vstack((x_cutpnts, x_cutpnts_siz))
            del itr8r

            itr8r = 0
            #for ww in ydim_gaps_b: #orig
            for ww in range(0,len(ydim_gaps_b)-1): # EJD edit 3/27/23 to fix index error #***
                ycut = ((ydim_gaps_b[itr8r] + ydim_gaps_e[itr8r])/2)
                ysiz = (ydim_gaps_e[itr8r]-ydim_gaps_b[itr8r])
                y_cutpnts.append(ycut)
                y_cutpnts_siz.append(ysiz)
                itr8r = itr8r+1
            y_cutpnts = np.vstack((y_cutpnts, y_cutpnts_siz))
            del itr8r

            # Iterate through clusters below the header looking for horizontal
            # conflicts. If one is found, add cut nearest to the rightmost edge
            # of the rightmost b_box in conflict. Rinse and repeat.. (iterating)
            # through each segment chunk individually untill no conflicts arise

            # rectangles given by: bL, tr
            # rectangle surrounding header segment
            header_seg = [[0, -header_ymin], [xdim, 0]]

            # initialize rectangular segment encompassing all below header
            seg_init = [[0, -ydim], [xdim, -header_ymin]]

            # Get list of clusters in header_seg and seg_init
            header_seg_cls = []
            seg_init_cls = []
            for xx in range(0, (cluster_mat.shape[1])):
                cl_ctrd = [cluster_mat[1][xx], -(cluster_mat[2][xx])]

                # Check if centroid falls in header region
                #hdr_chk = rctgle_chk(header_seg[0], header_seg[1], cl_ctrd)

                # Check whether centroid falls below header region
                non_hdr_chk = rctgle_chk(seg_init[0], seg_init[1], cl_ctrd)

                #if hdr_chk == True:
                if ((cl_ctrd[1]*-1) < header_ymin) == True:
                    header_seg_cls.append(round(cluster_mat[0][xx]))

                if non_hdr_chk == True:
                    seg_init_cls.append(round(cluster_mat[0][xx]))
            pause = ""

            clstr_readout = []  # Preallocate list for saving the order of the
            # clusters to be read out...

            # Pre-allocate space for saving cutpoints as needed in header and
            # body of slide

            header_cuts = []
            main_cuts = []
            
            # EJD: NEED TO REWORK DESCRIPTION BELOW!!
            # go through clusters in the header rectangle and check for
            # horizontal conflits.. if conflicts exist, iteratively add cuts
            # until horiz. conflicts are eliminated
            try: header_lsts
            except NameError: header_lsts = None
            
            if header_lsts is None:
                do_nothing = "nothing"
            else:
                header_lsts = None
            
            if len(header_seg_cls) > 0:
                header_lsts = []
                for zz in range(0, len(header_seg_cls)):
                    current_clust = cluster_mat[:, header_seg_cls[zz]]
                    lst = [round(current_clust[0])]
                    for ww in range(zz+1, len(header_seg_cls)):

                        cl_cmp_idx = header_seg_cls[ww]
                        clst_compare = cluster_mat[:, cl_cmp_idx]
                        cx = range(round(current_clust[4]), round(current_clust[3]))
                        cy = range(round(clst_compare[4]), round(clst_compare[3]))
                        xs = set(cx)
                        ol_check = xs.intersection(cy)
                        if len(ol_check) > 0:
                            lst.append(round(clst_compare[0]))

                    header_lsts.append(lst)
                
                # EJD placed these for debugging purposes. commented now..
                #if len(clusters)>1:
                #    print("header_lsts for len(clusters)>1:")
                #    print(header_lsts)
                #if len(clusters) == 1:
                #    print("header_lsts for len(clusters)==1:")
                #    print(header_lsts)
            
            # EJD: NEED TO REWORK DESCRIPTION BELOW!!
            # go through clusters in the main rectangle (all below header)
            # and check for horizontal conflits.. if conflicts exist,
            # iteratively add cuts until horiz. conflicts are eliminated
            try: main_lsts
            except NameError: main_lsts = None
            
            if main_lsts is None:
                do_nothing = "nothing"
            else:
                main_lsts = None
            
            if len(seg_init_cls) > 0:
                main_lsts = []
                for zz in range(0, len(seg_init_cls)):
                    current_clust = cluster_mat[:, seg_init_cls[zz]]
                    lst = [round(current_clust[0])]
                    for ww in range(zz+1, len(seg_init_cls)):

                        cl_cmp_idx = seg_init_cls[ww]
                        clst_compare = cluster_mat[:, cl_cmp_idx]
                        cx = range(round(current_clust[4]), round(current_clust[3]))
                        cy = range(round(clst_compare[4]), round(clst_compare[3]))
                        xs = set(cx)
                        ol_check = xs.intersection(cy)
                        if len(ol_check) > 0:
                            lst.append(round(clst_compare[0]))

                    main_lsts.append(lst)

            pause = "pause"
            
            # Iterate through main_lsts and combine lists which share =>1 cluster#
            # and eliminate duplicates...
            
            try: main_lsts
            except NameError: main_lsts = None
            
            if main_lsts is None:
                do_nothing = "nothing"
            else:
                main_lsts_nodups = []
                itr8r = 1
                fpass = 1;
                countr = 0;
                for zz in main_lsts:
                    l1 = set(zz)
                    
                    if itr8r >= len(main_lsts):
                        countr = 0
                        itr8r_3 = 0
                        for jj in main_lsts_nodups:
                            l2 = jj
                            lst_ol_check = l1.intersection(l2)
                            if len(lst_ol_check) > 0:
                                tmp_lst = copy.deepcopy(zz)
                                for gg in range(0, len(l2)):
                                    tmp_lst.append(l2[gg])
                                tmp_lst = set(tmp_lst)
                                tmp_lst = list(tmp_lst)
                                main_lsts_nodups[itr8r_3] = tmp_lst
                                countr = countr+1
                            else:
                                tmp_lst = copy.deepcopy(zz)
                            itr8r_3 = itr8r_3+1
                       
                        if countr == 0:
                            tmp_lst = copy.deepcopy(zz) #EJD ADDED 3/27/23 testing if this fixes issue.. #***
                            #np.disp("####################################")#EJD ADDED 3/27/23 testing if this fixes issue..
                            #np.disp("tmp list contents:")#EJD ADDED 3/27/23 testing if this fixes issue..
                            #np.disp(str(tmp_lst)) #EJD ADDED 3/27/23 testing if this fixes issue..
                            #np.disp("####################################")#EJD ADDED 3/27/23 testing if this fixes issue..
                            main_lsts_nodups.append(tmp_lst)
                            
                    for hh in range(itr8r, len(main_lsts)):
                        l2 = main_lsts[hh]
                        lst_ol_check = l1.intersection(l2)
                        
                        if len(lst_ol_check) > 0:
                            tmp_lst = copy.deepcopy(zz)
                            for gg in range(0, len(l2)):
                                tmp_lst.append(l2[gg])

                            tmp_lst = set(tmp_lst)
                            tmp_lst = list(tmp_lst)
                        else:
                            tmp_lst = copy.deepcopy(zz)
                            
                        if fpass == 1:
                            main_lsts_nodups.append(tmp_lst)
                            fpass = 0;
                        else:
                            itr8r_2 = 0
                            countr = 0;
                            for qq in main_lsts_nodups:
                                tset = set(qq)
                                chkr = tset.intersection(tmp_lst)
                                
                                if len(chkr) > 0:
                                    tset = list(tset)
                                    for ff in range(0, len(tmp_lst)):
                                        tset.append(tmp_lst[ff])
                                    tset = set(tset)
                                    tset = list(tset)
                                    main_lsts_nodups[itr8r_2] = tset
                                else:
                                    countr = countr+1
                                if countr == len(main_lsts_nodups):
                                    main_lsts_nodups.append(tmp_lst)
                                itr8r_2 = itr8r_2+1
                            countr = 0;                                   
                    itr8r = itr8r+1

            try: header_lsts
            except NameError: header_lsts = None            

            if header_lsts is None:
                do_nothing = "nothing"
            else:
                if len(clusters) > 1:
                    header_lsts_nodups = []
                    itr8r = 1
                    fpass = 1;
                    countr = 0;
                    for zz in header_lsts:
                        l1 = set(zz)
                        if itr8r >= len(header_lsts):
                            countr = 0
                            itr8r_3 = 0
                            for jj in header_lsts_nodups:
                                l2 = jj
                                lst_ol_check = l1.intersection(l2)
                                if len(lst_ol_check) > 0:
                                    tmp_lst = copy.deepcopy(zz)
                                    for gg in range(0, len(l2)):
                                        tmp_lst.append(l2[gg])
                                    tmp_lst = set(tmp_lst)
                                    tmp_lst = list(tmp_lst)
                                    header_lsts_nodups[itr8r_3] = tmp_lst
                                    countr = countr+1
                                else:
                                    tmp_lst = copy.deepcopy(zz)
                                itr8r_3 = itr8r_3+1
                            if countr == 0:
                                header_lsts_nodups.append(tmp_lst)
                                
                        for hh in range(itr8r, len(header_lsts)):
                            l2 = header_lsts[hh]
                            lst_ol_check = l1.intersection(l2)
                            
                            if len(lst_ol_check) > 0:
                                tmp_lst = copy.deepcopy(zz)
                                for gg in range(0, len(l2)):
                                    tmp_lst.append(l2[gg])
    
                                tmp_lst = set(tmp_lst)
                                tmp_lst = list(tmp_lst)
                            else:
                                tmp_lst = copy.deepcopy(zz)
                                
                            if fpass == 1:
                                fpass = 0;
                                header_lsts_nodups.append(tmp_lst)
                                    
                            else:
                                itr8r_2 = 0
                                countr = 0;
                                for qq in header_lsts_nodups:
                                    tset = set(qq)
                                    chkr = tset.intersection(tmp_lst)
                                    
                                    if len(chkr) > 0:
                                        tset = list(tset)
                                        for ff in range(0, len(tmp_lst)):
                                            tset.append(tmp_lst[ff])
                                        tset = set(tset)
                                        tset = list(tset)
                                        header_lsts_nodups[itr8r_2] = tset
                                    else:
                                        countr = countr+1
                                    if countr == len(header_lsts_nodups):
                                        header_lsts_nodups.append(tmp_lst)
                                    itr8r_2 = itr8r_2+1
                                countr = 0;                                   
                        itr8r = itr8r+1
                # If there is header text but only one cluster, no need to check for dupes.. theres only one..
                if len(clusters) == 1:
                    header_lsts_nodups=copy.deepcopy(header_lsts)
            # A) For both the header and main clusters, sort the clusters in each
            # list by centroid from top to bottom and from left to right.
            
            # B) Then get the centroid for each list (in header and main.. nodups)
            # and compute the centroid for all clusters on the list. Sort these 
            # from left to right
            
            
            try: main_lsts
            except NameError: main_lsts = None
            
            if main_lsts is None:
                do_nothing = "nothing"
            else: 
                # A)
                main_lsts_nodups_coords_srt =[];
                for tt in main_lsts_nodups:
                    tmp_lst = copy.deepcopy(tt)
                    tmp_lst2 = np.zeros((2, len(tmp_lst)), dtype=np.uint8)
                    tmp_lst = np.vstack((tmp_lst,tmp_lst2))
                    
                    for ss in range(0,tmp_lst.shape[1]):
                        # tmp_lst[1][ss] = round(cluster_mat[1][tmp_lst[0][ss]], -1)
                        # tmp_lst[2][ss] = round(cluster_mat[2][tmp_lst[0][ss]], -1)
                        tmp_lst[1][ss] = round(cluster_mat[1][tmp_lst[0][ss]], -1)
                        tmp_lst[2][ss] = round(cluster_mat[2][tmp_lst[0][ss]], -1)
                    
                    tmp_lst = tmp_lst[:, tmp_lst[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                    tmp_lst = tmp_lst[:, tmp_lst[1, :].argsort(kind='mergesort')] # Then sort on x coordinate (L-->R)
                    tmp_lst = tmp_lst[:, tmp_lst[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                    
                    main_lsts_nodups_coords_srt.append(tmp_lst)
                    
                        
                # B)   
                main_lst_cntrds_x = []
                main_lst_cntrds_y = []
                for rr in main_lsts_nodups_coords_srt:
                    x_aveTmp = round(np.mean(rr[1][:]))
                    y_aveTmp = round(np.mean(rr[2][:]))
                    main_lst_cntrds_x.append(x_aveTmp)
                    main_lst_cntrds_y.append(y_aveTmp)
    
                main_lst_cntrds_lbls = list(range(0,len(main_lst_cntrds_y))) # saves indices of main_lsts_nodups_coords_srt for later/sorting..
                main_lst_cntrds = np.vstack((main_lst_cntrds_lbls,main_lst_cntrds_x))
                main_lst_cntrds = np.vstack((main_lst_cntrds,main_lst_cntrds_y))
                
                main_lst_cntrds = main_lst_cntrds[:, main_lst_cntrds[1, :].argsort()]
                
                pause = "pause"
            
            try: header_lsts
            except NameError: header_lsts = None            

            if header_lsts is None:
                do_nothing = "nothing"
            else:
                # A)
                header_lsts_nodups_coords_srt =[];
                
                #EJD placed these for debugging... commented now... 
                #if len(clusters) == 1:
                #    print("header_lsts_nodups:")
                #    print(header_lsts_nodups)
                    
                for tt in header_lsts_nodups:
                    tmp_lst = copy.deepcopy(tt)
                    tmp_lst2 = np.zeros((2, len(tmp_lst)), dtype=np.uint8)
                    tmp_lst = np.vstack((tmp_lst,tmp_lst2))
                    
                    
                    
                    for ss in range(0,tmp_lst.shape[1]):
                        tmp_lst[1][ss] = round(cluster_mat[1][tmp_lst[0][ss]], -1)
                        tmp_lst[2][ss] = round(cluster_mat[2][tmp_lst[0][ss]], -1)
                        
                    tmp_lst = tmp_lst[:, tmp_lst[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                    tmp_lst = tmp_lst[:, tmp_lst[1, :].argsort(kind='mergesort')] # Then sort on x coordinate (L-->R)
                    tmp_lst = tmp_lst[:, tmp_lst[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                    
                    header_lsts_nodups_coords_srt.append(tmp_lst)
                
                #EJD placed these for debugging... commented now...    
                #if len(clusters) == 1:
                #    print("header_lsts_nodups_coords_srt:")
                #    print(header_lsts_nodups_coords_srt)
                    
                    
                
                # B)
                header_lst_cntrds_x = []
                header_lst_cntrds_y = []
                for rr in header_lsts_nodups_coords_srt:
                    x_aveTmp = round(np.mean(rr[1][:]))
                    y_aveTmp = round(np.mean(rr[2][:]))
                    header_lst_cntrds_x.append(x_aveTmp)
                    header_lst_cntrds_y.append(y_aveTmp)
    
                header_lst_cntrds_lbls = list(range(0,len(header_lst_cntrds_y))) # saves indices of main_lsts_nodups_coords_srt for later/sorting..
                header_lst_cntrds = np.vstack((header_lst_cntrds_lbls,header_lst_cntrds_x))
                header_lst_cntrds = np.vstack((header_lst_cntrds,header_lst_cntrds_y))
                
                header_lst_cntrds = header_lst_cntrds[:, header_lst_cntrds[1, :].argsort()]
                
                pause = "pause"
            
            
            # READ IF YOU WANT TO KNOW HOW ON EARTH THE ABOVE CODE/STUFF
            # CREATED ABOVE IS USED TO WRITE OUT TEXT CLUSTER IMAGES IN AN
            # ORDER WHICH KEEPS TEXT DETECTED IN EACH IMAGE FRAME GROUPED IN
            # A "HUMAN READABLE" MANNER..
            # ****************************************************************
            # ****************************************************************
            #
            # Now: We will use the first row in header_lst_cntrds and 
            # main_lst_cntrds to tell us what order to read out the arrays 
            # ("clusters of clusters") contained in 
            # header_lsts_nodups_coords_srt and main_lsts_nodups_coords_srt.
            # These values (in first row of header_lst_cntrds/main_lst_cntrds)
            # represent list indices of the arrays contained within these lists 
            # (header_lsts_nodups_coords_srt and main_lsts_nodups_coords_srt)
            # Reading out arrays positioned at these indices in this prescried
            # order ensures the "clusters of clusters" are read out L->R and 
            # top->bottom.)
            #
            # The manner in which the clusters in the "clusters of clusters"
            # are clustered is easy(ish) to describe but was somewhat complicated
            # to code.. The rule was this: all clusters with overlapping
            # x-coordinate values were grouped together. If a cluster A shared 
            # x-coordinates with another cluster B, and B shared x coordinates
            # with some other cluster C (which, lets say, doesn't overlap with 
            # A), A, B, C all group together. Useful analogy (at least for me): 
            # Imagine clusters are globs of playdoh arranged on a page. 
            # Each glob gets connected to any glob directly above or below it 
            # with a toothpick. Each independent mass of interconnected playdoh 
            # /toothpick globs is a "cluster of clusters."
            #
            # The values in the top rows of each array listed in 
            # header_lsts_nodups_coords_srt and main_lsts_nodups_coords_srt
            # represent cluster numbers (ie indices within the "clusters" list
            # variable). These should also be read out in the order listed as
            # they have been sorted (like header_lst_cntrds and main_lst_cntrds)
            # such that clusters are read out L->R/Top->Bottom.
            #
            # The "clusters" variable is a list. Each index inside that list 
            # is a list of np arrays. These np arrays are coordinates of the 
            # the bounding boxes (or more aptly.. bounding "snakes") created by
            # MMOCR's TextSnake text detector. Each index in "clusters" is a
            # list of bounding boxes which were clustered using the numpy 
            # agglomerative clustering algorithm. (this was done to keep 
            # lines of text in close spatial proximity (and highly likely to 
            # be intended to be read as a group) bound together. (Note: not all
            # clusters contain >1 bounding box and are therefore sometimes a 
            # list of length=1)
            #
            # These bounding boxes/snakes are then used to mask the text
            # contained within them (Using opencv2's "fillPoly" function and
            # some numpy wizardry). This crops away everything outside 
            # the bounding snakes leaving only the text on a white backround.
            # (exactly what Tesseract likes..) Bounding boxes grouped together
            # in each cluster are applied together in the same mask. The image
            # arrays are then written out as .jpeg files which are named 
            # systematically with tags (000->NNN) such that they can be easily
            # sorted/kept in order and fed through Tesseract.
            #
            # (Image names also indicate the video image frames/timepoints the
            # images came from to keep track of when/where in each lecture they
            # come from..)
            #
            # Tesseract then recognizes the text in each image (000->NNN) and
            # writes it out as a ?line? (double check) of test in the .txt
            # output. Seperate frames are also labeled w/in the txt output.
            #
            # EJD NOTE/UPDATE (12/30/22):
            # At this point, I have essentially dropped the idea of having tesseract
            # be the final reader of the text. Was the original intention, but the
            # more I learned/this project (data scraping from video lectures) 
            # developed, the more it made sense to stick with a single detection program.
            # this ended up being MMOCR. This program is still integral
            # to the project. However, the real utility/role it plays now is
            # in clustering. In short, his program clusters the text outputs from 
            # mmocr spatially based on their bounding bo coordinates and sequences
            # them such that they can be read out in a human readable order.
            #
            # In this way, the primary "output" is no longer a directory full of images
            # to be read in by tesseract or some other program. Instead, the "primary
            # output" is a hierarchically organized "list-of-lists" structure called
            # txt_out_final containing, at the highest level, a list for each frame.
            #
            # a detailed explanation of the output structure is pasted below:
            # (Ethan pasted this from lecxr_text_v2.py, the "main program" for
            # video lecture text extraction.. he's only just gotten
            # around to integrating all of the pieces-parts of the whole
            # data-scraping pipeline together..)
            #
            # Text extracted from each of the unique image frames: currenltly stored in "txt_out_final"
            #    (sry for the lengthy description.. but if you need to understand the contents of txt_out, you'll need to understand all this..)
            #       * txt_out_final is a list of nested lists
            #       * each of the "highest level" lists is a video frame. (lets call these "frame_lists" in this explanation..)
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
            # ****************************************************************
            # ****************************************************************
            
            txt_out = [];
            try: header_lsts
            except NameError: header_lsts = None            

            if header_lsts is None:
                do_nothing = "nothing"
            else:
                arb_it = 0
                for ee in range(0,(header_lst_cntrds.shape[1])):
                    clstr_o_clstrs = header_lsts_nodups_coords_srt[header_lst_cntrds[0][ee]]
                    tmp_list1 = []
                    for qq in range(0,(clstr_o_clstrs.shape[1])):
                        clst = clstr_o_clstrs[0][qq]
                        clist = clusters_txt[round(clst)]
                        # Adding this step to ensure items w/in agglom clusters are read out top->bottom , L->R
                        clist_coord = clusters[round(clst)]
                        
                        clist_coord_tmpSrt = np.zeros((3,len(clist_coord)),dtype=object);
                        itr8r = 0;
                        for dd in clist_coord:
                            xtmp =[];
                            ytmp = [];
                            for hh in dd:
                                xtmp.append(hh[0])
                                ytmp.append(hh[1])
                            xtmpMin = min(xtmp)
                            ytmpMin = min(ytmp)
                            clist_coord_tmpSrt[0][itr8r] = itr8r
                            clist_coord_tmpSrt[1][itr8r] = round(xtmpMin,-1)
                            clist_coord_tmpSrt[2][itr8r] = round(ytmpMin,-1)
                            itr8r = itr8r+1
                        
                        
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[1, :].argsort(kind='mergesort')] # Then sort on x coordinate (L-->R)
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                        tmp_list2 = []
                        itr8r = 0
                        for bb in clist:
                            tmp_list2.append(clist[clist_coord_tmpSrt[0][itr8r]])
                            itr8r = itr8r+1
                        del bb

    
                        tmp_list1.append(tmp_list2)
                        arb_it = arb_it+1
                    txt_out.append(tmp_list1)
                    itr8r=0
            
            
            if header_lsts is None:
                arb_it = 0   
            else:
                do_nothing = "nothing"
            
            try: main_lsts
            except NameError: main_lsts = None
            
            if main_lsts is None:
                do_nothing = "nothing"
            else: 
                for ee in range(0,(main_lst_cntrds.shape[1])):
                    clstr_o_clstrs = main_lsts_nodups_coords_srt[main_lst_cntrds[0][ee]]
                    tmp_list1 = []
                    for qq in range(0,(clstr_o_clstrs.shape[1])):
                        clst = clstr_o_clstrs[0][qq]
                        clist = clusters_txt[round(clst)]
                        
                        # Adding this step to ensure items w/in agglom clusters are read out top->bottom , L->R
                        clist_coord = clusters[round(clst)]
                        
                        clist_coord_tmpSrt = np.zeros((3,len(clist_coord)),dtype=object);
                        itr8r = 0;
                        for dd in clist_coord:
                            xtmp =[];
                            ytmp = [];
                            for hh in dd:
                                xtmp.append(hh[0])
                                ytmp.append(hh[1])
                            xtmpMin = min(xtmp)
                            ytmpMin = min(ytmp)
                            clist_coord_tmpSrt[0][itr8r] = itr8r
                            clist_coord_tmpSrt[1][itr8r] = round(xtmpMin,-1)
                            clist_coord_tmpSrt[2][itr8r] = round(ytmpMin,-1)
                            itr8r = itr8r+1
                        
                        
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[1, :].argsort(kind='mergesort')] # Then sort on x coordinate (L-->R)
                        clist_coord_tmpSrt = clist_coord_tmpSrt[:, clist_coord_tmpSrt[2, :].argsort()] # Sort on Y coordinate (top-->bottom)
                        tmp_list2 = []
                        itr8r = 0
                        for bb in clist:
                            tmp_list2.append(clist[clist_coord_tmpSrt[0][itr8r]])
                            itr8r = itr8r+1
                        del bb
                        tmp_list1.append(tmp_list2)
                        arb_it = arb_it+1
                    txt_out.append(tmp_list1)

            pause = "pause"
            txt_out_final.append(txt_out)


            ###################################################################
            # UNCOMMENTING THE SECTION BELOW WILL MAKE THIS MODULE PRINT OUT
            # MASKED IMAGES OF EACH BOUNDING BOX IN THE ORDER IN WHICH THE 
            # TEXT WITHIN THEM ARE READ OUT INTO THE .TXT OUTPUT
            ###################################################################
            
            if not os.path.exists("cluster_viz"):
                os.mkdir("cluster_viz")
            os.chdir("cluster_viz")
            
            try: header_lsts
            except NameError: header_lsts = None            

            if header_lsts is None:
                do_nothing = "nothing"
            else:
                arb_it = 0
                for ee in range(0,(header_lst_cntrds.shape[1])):
                    clstr_o_clstrs = header_lsts_nodups_coords_srt[header_lst_cntrds[0][ee]]
                    for qq in range(0,(clstr_o_clstrs.shape[1])):
                        clst = clstr_o_clstrs[0][qq]
                        mask2 = np.zeros((ydim, xdim), dtype=np.uint8)
                        clist = clusters[round(clst)]
    
                        for bb in clist:
                            mask2 = cv2.fillPoly(mask2, np.int32([bb]), 1)
                        del bb
                        mask2_inv = np.logical_not(mask2)
                        mask2_inv = np.multiply(mask2_inv, 1)
                        mask2_inv = mask2_inv*255
                        mask2_str_base = os.path.splitext(im_list[ii])[0]
                        mask2_str = mask2_str_base+"_" + \
                            str(arb_it).zfill(3)+"_txt_mask_apld.jpg"
                        masked_im2 = (img*mask2)+mask2_inv
                        masked_im_0b = (img*mask2)
    
                        cv2.imwrite(mask2_str, masked_im2)
    
                        arb_it = arb_it+1
            
            
            if header_lsts is None:
                arb_it = 0   
            else:
                do_nothing = "nothing"
            
            try: main_lsts
            except NameError: main_lsts = None
            
            if main_lsts is None:
                do_nothing = "nothing"
            else: 
                for ee in range(0,(main_lst_cntrds.shape[1])):
                    clstr_o_clstrs = main_lsts_nodups_coords_srt[main_lst_cntrds[0][ee]]
                    for qq in range(0,(clstr_o_clstrs.shape[1])):
                        clst = clstr_o_clstrs[0][qq]
                        mask2 = np.zeros((ydim, xdim), dtype=np.uint8)
                        clist = clusters[round(clst)]
    
                        for bb in clist:
                            mask2 = cv2.fillPoly(mask2, np.int32([bb]), 1)
                        del bb
                        mask2_inv = np.logical_not(mask2)
                        mask2_inv = np.multiply(mask2_inv, 1)
                        mask2_inv = mask2_inv*255
                        mask2_str_base = os.path.splitext(im_list[ii])[0]
                        mask2_str = mask2_str_base+"_" + \
                            str(arb_it).zfill(3)+"_txt_mask_apld.jpg"
                        masked_im2 = (img*mask2)+mask2_inv
                        masked_im_0b = (img*mask2)
    
                        cv2.imwrite(mask2_str, masked_im2)
    
                        arb_it = arb_it+1
 
            #pause = "pause"
            os.chdir(im_dir)
            ##################################################################
            ##################################################################
        else:
            if len(contour_list) == 1:
                txt_out = [[str(j_data['result'][0]['text'])]]
                txt_out_final.append([txt_out])
                
            if len(contour_list) == 0:
                txt_out = [[""]]
                txt_out_final.append([txt_out])
            
            
            
    return txt_out_final


            # Find the number of vertical cuts necessary to get 1 text cluster
            # per "row" in the image

            # Define a row as a proportion of the page? or a multiple the cluster
            # height?

            # pgRowH =

            #contour_clust_dict = dict(zip(a_set, clusters))

        # else:
        #     aset_dummy ={"0"}
        #     clusters_dummy = [contours]
        #     contour_clust_dict = dict(zip(aset_dummy, clusters_dummy))

            
            # print out seperate cluster images labeling them 0-n based on cluster centroid sorted top-to-bottom and left-to-right
            # im_copies = [];
            # itr = 0;
            # mask3 = np.zeros((ydim,xdim), dtype=np.uint8);
            # for uu in level2_boxes:
            #     im_cpy = copy.deepcopy(masked_im2)
            #     msk_cpy = copy.deepcopy(mask3)
            #     im_base = os.path.splitext(im_list[ii])[0]
            #     im_str = im_base+"_level2boxNum_"+str(itr).zfill(3)+".jpeg"

            #     cv2.rectangle(msk_cpy, (level2_boxes[itr][0][0], level2_boxes[itr][0][1]), (level2_boxes[itr][1][0], level2_boxes[itr][1][1]), (1, 1, 1), -1)

            #     msk_cpy_inv = np.logical_not(msk_cpy)
            #     msk_cpy_inv = np.multiply(msk_cpy_inv, 1)
            #     msk_cpy_inv = msk_cpy_inv*255
            #     masked_im_msk_cpy = (img*msk_cpy)+msk_cpy_inv

            #     #cv2.rectangle(im_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     cv2.imwrite(im_str, masked_im_msk_cpy)
            #     itr = itr+1










            #*******************************************************
            # THIS IS THE OG READOUT BLOCK! EJD COMMENTED TO TRY NEW 
            # for ee in cluster_labels_sorted:
            #     mask2 = np.zeros((ydim, xdim), dtype=np.uint8)
            #     clist = clusters[round(ee)]

            #     for bb in clist:
            #         mask2 = cv2.fillPoly(mask2, np.int32([bb]), 1)
            #     del bb

            #     #cv2.fillPoly(mask2, np.int32([contour_clust_dict[ee]]),1)
            #     mask2_inv = np.logical_not(mask2)
            #     mask2_inv = np.multiply(mask2_inv, 1)
            #     mask2_inv = mask2_inv*255
            #     mask2_str_base = os.path.splitext(im_list[ii])[0]
            #     mask2_str = mask2_str_base+"_" + \
            #         str(arb_it).zfill(3)+"_txt_mask_apld.jpg"
            #     masked_im2 = (img*mask2)+mask2_inv
            #     masked_im_0b = (img*mask2)

            #     cv2.imwrite(mask2_str, masked_im2)

            #     arb_it = arb_it+1
            # del ee
            #*******************************************************
            # THIS IS THE OG READOUT BLOCK! EJD COMMENTED TO TRY NEW 









        # # for ii in range(0,len(contour_list)-1):
        # #     lst_num = labels(ii)

        # mask_inv = np.logical_not(mask)
        # mask_inv = np.multiply(mask_inv, 1)
        # mask_inv = mask_inv*255
        # mask_str_base = os.path.splitext(im_list[ii])[0]
        # mask_str = mask_str_base+"_txt_mask.jpg"
        # maskinv_str = mask_str_base+"_txt_mask_inverted.jpg"

        # #EJD COMMENTED.. WASN'T USING
        # mask_apld_str = mask_str_base+"_all_txt_masks.jpg"

        # #cv2.imwrite(mask_str, mask)
        # #cv2.imwrite(maskinv_str, mask_inv)
        # masked_im = (img*mask)+mask_inv
        # cv2.imwrite(mask_apld_str, masked_im)

        #pause = "pause"
        #contours_stacked = np.vstack(contour_list)
        #pause = "pause"

        # txt_image = cv2.imread(mask_str_base+"_txt_mask_apld.jpg",0)
        # mask3 = np.zeros((ydim,xdim), dtype=np.uint8)
        # contour_stacked_master = []
        # for shapes in contour_list:
        #     #s = shapes
        #     int_list = [shapes];

        #     for shapes2 in contour_list:
        #         intersect = contourIntersect(mask3, shapes, shapes2)
        #         if intersect == 1:
        #             int_list.append(shapes2)

        #     contours_stacked = np.vstack(int_list)

        #     contour_stacked_master.append(contours_stacked)
        #     del contours_stacked
        #     del int_list

        # contour_stacked_master2 = []
        # for shapes3 in contour_stacked_master:
        #     #s = shapes
        #     int_list2 = [shapes3];

        #     for shapes4 in contour_list:
        #         intersect = contourIntersect(mask3, shapes3, shapes4)
        #         if intersect == 1:
        #             int_list2.append(shapes4)
        #     contours_stacked = np.vstack(int_list2)

        #     contour_stacked_master2.append(contours_stacked)
        #     del contours_stacked
        #     del int_list2

        # #contour_stacked_master_unique = list(set(contour_stacked_master))

        # count = 1
        # for shapes5 in contour_stacked_master2:
        #     mask2 = np.zeros((ydim,xdim), dtype=np.uint8)
        #     mask2 = cv2.fillPoly(mask2, np.int32([shapes5]),1)
        #     mask2_inv = np.logical_not(mask2)
        #     mask2_inv = np.multiply(mask2_inv, 1)
        #     mask2_inv = mask2_inv*255
        #     mask_apld_str2 = mask_str_base+"_txt_mask_apld_txtn_"+str(count).zfill(5)+".jpg"
        #     masked_im2 = (img*mask2)+mask2_inv
        #     cv2.imwrite(mask_apld_str2, masked_im2)
        #     count = count+1


def bbox_txtMask_snek_tsrc(im_dir, txt_dir, im_tag, outbase):  # ****
    import cv2
    import os
    import glob
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    import pandas as pd
    import numpy as np
    #from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as sch
    from operator import itemgetter
    import copy
    from sklearn.cluster import OPTICS, cluster_optics_dbscan
    import matplotlib.gridspec as gridspec
    import pytesseract
    from pytesseract import Output
    from fpdf import FPDF

    os.chdir(im_dir)
    # get list of images containing tag
    im_list = glob.glob(im_tag)
    im_list.sort()  # make sure they are sorted in ascending order
    n_im = len(im_list)

    # os.chdir(txt_dir)
    # get list of text files containing tag
    #txt_list = glob.glob(txt_tag)
    # txt_list.sort() # make sure they are sorted in ascending order
    #n_txt = len(txt_list)

    # os.chdir(im_dir)

    # Get image dimensions
    img4dims = cv2.imread(im_list[0], 0)
    ydim, xdim = img4dims.shape

    for ii in range(0, n_im):
        img = cv2.imread(im_list[ii], 0)
        #mask = np.zeros((ydim,xdim,zdim), np.uint8)
        mask = np.zeros((ydim, xdim), dtype=np.uint8)

        #j_file = txt_list[ii]
        # Opening JSON file
        #j_file = open(txt_dir+j_file)

        # returns JSON object as
        # a dictionary
        #j_data = json.load(j_file)

        # Iterating through the json
        # list
        #size_obj = len(j_data['boundary_result'])

        # contour_list = []
        # centroid_lst =[]
        # centroid_lst_wh = []
        # clust_heights = []
        # for i in range (0,size_obj):
        #     shape_coord = j_data['boundary_result'][i]
        #     shape = []
        #     xpt = int(0)
        #     ypt = int(1)
        #     npairs = int((len(shape_coord)-1)/2)
        #     for uu in range(0, npairs-1):
        #         xpoint = int(round(shape_coord[xpt]))
        #         ypoint = round(int(shape_coord[ypt]))
        #         point = [xpoint,ypoint]
        #         shape.append(point)
        #         xpt = xpt+2
        #         ypt = ypt+2

        #     contours = np.array(shape)
        #     x_shp,y_shp,w_shp,h_shp = cv2.boundingRect(contours)
        #     M = cv2.moments(contours)
        #     cX = int(M["m10"] / M["m00"])
        #     cY= int(M["m01"] / M["m00"])
        #     centroid = [cX,cY]
        #     centroid_wh = [cX,cY,h_shp]
        #     clust_heights.append(h_shp)

        #     centroid_lst.append(centroid)
        #     centroid_lst_wh.append(centroid_wh)
        #     contour_list.append(contours)
        #     mask = cv2.fillPoly(mask, np.int32([contours]),1)

        # mask_inv = np.logical_not(mask)
        # mask_inv = np.multiply(mask_inv, 1)
        # mask_inv = mask_inv*255
        # masked_im = (img*mask)+mask_inv
        # masked_im2 = masked_im.astype(np.uint8)

        # masked_im3 = copy.deepcopy(masked_im2)

        # # masked_im4 = copy.deepcopy(masked_im2)
        # # masked_im5 = copy.deepcopy(masked_im2)
        # # masked_im6 = copy.deepcopy(masked_im2)

        # d = pytesseract.image_to_data(masked_im2, output_type=Output.DICT)
        # # print(d.keys()) #uncomment to print out other items in 'd.keys'
        # n_boxes = len(d['level'])
        # lmax = max(d['level']);
        # lmin = min(d['level']);

        # # Note: Lowest level appears to just be the entire image..
        # #       Highest level is individual words
        # #       Mid Levels are probably the "sub groups" we're looking for?
        # #
        # #       After looking at each level.. it looks like things progress like above from
        # #       whole page to finer and finer divisions as level increases..
        # #       ** for whatever reason 2&3 appear the same in this image..
        # #       **HYPOTHESIS: Level 2 is the "coarsest" division of blocks that
        # #       is not the whole page**
        # level_boxes = []
        # for i in range(n_boxes):
        #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

        #     #if d['level'][i] == lmin:
        #     #    cv2.rectangle(masked_im3, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #     #if d['level'][i] == lmax:
        #     #    cv2.rectangle(masked_im4, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #     level_boxes.append([[x, y], [x + w, y + h]])
        #     cv2.rectangle(masked_im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #     #if d['level'][i] == 3:
        #     #    cv2.rectangle(masked_im5, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #     #if d['level'][i] == 4:
        #     #    cv2.rectangle(masked_im6, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # pause = "pause"
        # #cv2.imwrite("test_L1.jpeg",masked_im3)
        # #cv2.imwrite("test_L5.jpeg",masked_im4)

        # Get a searchable PDF

        # cv2.imwrite(os.path.splitext(im_list[ii])[0]+'_tstpdf.jpeg',masked_im3)
        tmp_base = os.path.splitext(im_list[ii])[0]
        with open(tmp_base+'.pdf', 'w+b') as f1:

            pdf = pytesseract.image_to_pdf_or_hocr(
                im_list[ii], extension='pdf')

            f1.write(pdf)  # pdf type is bytes by default

            # UNCOMMENT TO VIEW THE BOXES...
            # cv2.imwrite(os.path.splitext(im_list[ii])[0]+"_AllLevel_boxes.jpeg",masked_im2)

            # #cv2.imwrite("test_L3.jpeg",masked_im5)
            # #cv2.imwrite("test_L4.jpeg",masked_im6)
            #     # factor = 0.15

            #     # height = abs(box_coord[2]-box_coord[6])
            #     # h_expd = height*(1+factor)
            #     # h_add = round((abs(height - h_expd)/2))

            #     # width = abs(box_coord[3]-box_coord[7])
            #     # w_expd = width*(1+factor)
            #     # w_add = round((abs(width - w_expd)/2))

            #     # #draw box as rectangle in mask image
            #     # cv2.rectangle(mask,(box_coord[2]+h_add,box_coord[3]-w_add),(box_coord[6]-h_add,box_coord[7]+w_add),(1,1,1),-1)
            #     # #plt.imshow(mask, cmap='plasma')

            # #plt.imshow(mask, cmap='plasma')
            # #plt.show()
            # #mask_inv = cv2.bitwise_not(mask)

            # #make dendrogram of clustering for visualization

            # #if len(contour_list)>1:

            #     #dendrogram = sch.dendrogram(sch.linkage(centroid_lst,affinity='euclidean', method='single'))

            #     #Fit the clustering model and get labels for centroid list
            #     #d_thr = 0.1*(ydim)

            #     # EJD ALTERNATE CLUSTERING METHOD ATTEMPTS:
            #     # model = OPTICS(min_samples=1, xi=0.05, min_cluster_size=0.05)

            #     # model.fit(centroid_lst)
            #     # labels = model.labels_

            #     # from sklearn.cluster import DBSCAN
            #     # model = DBSCAN(eps=0.3, min_samples=2).fit(centroid_lst)
            #     # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
            #     # core_samples_mask[model.core_sample_indices_] = True
            #     # labels = model.labels_

            #     #ORIG CLUSTERING MODEL!!
            #     #ORIG CLUSTERING MODEL!!

            #     #d_thr = 2*(np.mean(clust_heights)) # commented

            #     d_thr = ClustThr_factor*(np.mean(clust_heights))

            #     model = AgglomerativeClustering(n_clusters=None,distance_threshold=d_thr,affinity=affinity, linkage=linkage)

            #     model.fit(centroid_lst)
            #     #model.fit(centroid_lst_wh)
            #     labels = model.labels_
            #     #ORIG CLUSTERING MODEL!!
            #     #ORIG CLUSTERING MODEL!!

            #     labels_lst = labels.tolist()
            #     for nn in range(0,len(labels_lst)):
            #         labels_lst[nn] = str(labels_lst[nn])

            #     # Get number of unique clusters
            #     a_list = labels_lst
            #     a_set = set(a_list)
            #     n_uVals = len(a_set)

            #     clLabeled_contours = np.vstack((labels_lst,contour_list))

            #     #lbl_cntrs_cntroids = np.vstack((labels_lst,contour_list,centroid_lst))
            #     #lbl_cntrs_cntroids.sort(0)

            #     clusters = []
            #     for yy in a_set:
            #         c_string = "cluster_"+yy
            #         cmd_str1 = c_string+"=[]"
            #         exec(cmd_str1)
            #         cmd_str2 = "clusters.append("+c_string+")"
            #         exec(cmd_str2)
            #         del yy

            #     cluster_centroids = copy.deepcopy(clusters)

            #     for bb in range(0,(clLabeled_contours.shape[1])):
            #         cl_idx = clLabeled_contours[0,bb]
            #         clusters[int(cl_idx)].append(clLabeled_contours[1,bb])
            #         cluster_centroids[int(cl_idx)].append(centroid_lst[bb])
            #     del bb

            #     # Get the average centroid coordinate for text boxes in each cluster (ie get the cluster centroid)
            #     ave_cluster_centroid_x =[]
            #     ave_cluster_centroid_y =[]
            #     ave_cluster_centroids = []
            #     for bb in cluster_centroids:
            #         xlist = []
            #         ylist = []
            #         for yy in bb:
            #             xvalue = yy[0]
            #             xlist.append(xvalue)
            #             yvalue = yy[1]
            #             ylist.append(yvalue)
            #         x_ave = np.mean(xlist)
            #         y_ave = np.mean(ylist)
            #         mean_cntrd = [x_ave, y_ave]
            #         ave_cluster_centroid_x.append(x_ave)
            #         ave_cluster_centroid_y.append(y_ave)
            #         ave_cluster_centroids.append(mean_cntrd)
            #     del bb
            #     del yy
            #     #cl_labels_cntrds = list(a_set)
            #     cl_labels_cntrds = list(range(0,n_uVals))
            #     ave_cluster_centroid_y_sort = np.vstack((copy.deepcopy(cl_labels_cntrds),copy.deepcopy(ave_cluster_centroid_y)))
            #     ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:, ave_cluster_centroid_y_sort[1, :].argsort()]
            #     #cluster_centroids_labeled = np.vstack((cl_labels_cntrds,ave_cluster_centroid_x,ave_cluster_centroid_y))

            #     #cluster_centroids_labeled = cluster_centroids_labeled[:, cluster_centroids_labeled[2, :].argsort()]
            #     #cluster_centroids_labeled = cluster_centroids_labeled[:, cluster_centroids_labeled[2, :].argsort(kind='mergesort')]
            #     #pause = "pause"

            #     cluster_labels_sorted = ave_cluster_centroid_y_sort[0]

            #     arb_it = 0
            #     # print out seperate cluster images labeling them 0-n based on cluster centroid sorted top-to-bottom and left-to-right

            #     itr = 0;
            #     mask3 = np.zeros((ydim,xdim), dtype=np.uint8);
            #     for uu in level_boxes:
            #         im_cpy = copy.deepcopy(masked_im3)
            #         msk_cpy = copy.deepcopy(mask3)
            #         im_base = os.path.splitext(im_list[ii])[0]
            #         im_str = im_base+"_level2boxNum_"+str(itr).zfill(3)+".jpeg"

            #         cv2.rectangle(msk_cpy, (level_boxes[itr][0][0], level_boxes[itr][0][1]), (level_boxes[itr][1][0], level_boxes[itr][1][1]), (1, 1, 1), -1)

            #         msk_cpy_inv = np.logical_not(msk_cpy)
            #         msk_cpy_inv = np.multiply(msk_cpy_inv, 1)
            #         msk_cpy_inv = msk_cpy_inv*255
            #         masked_im_msk_cpy = (im_cpy*msk_cpy)+msk_cpy_inv

            #         #cv2.rectangle(im_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #         cv2.imwrite(im_str, masked_im_msk_cpy)
            #         itr = itr+1
    pause = ""


def ind_bbox_txtMask_snek(im_dir, txt_dir, im_tag, txt_tag):
    import cv2
    import os
    import glob
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    os.chdir(im_dir)
    # get list of images containing tag
    im_list = glob.glob(im_tag)
    im_list.sort()  # make sure they are sorted in ascending order
    n_im = len(im_list)

    os.chdir(txt_dir)
    # get list of images containing tag
    txt_list = glob.glob(txt_tag)
    txt_list.sort()  # make sure they are sorted in ascending order
    n_txt = len(txt_list)

    os.chdir(im_dir)

    # Get image dimensions
    img4dims = cv2.imread(im_list[0], 0)
    ydim, xdim = img4dims.shape

    for ii in range(0, n_im):
        img = cv2.imread(im_list[ii], 0)
        #mask = np.zeros((ydim,xdim,zdim), np.uint8)
        #mask = np.zeros((ydim,xdim), dtype=np.uint8)

        j_file = txt_list[ii]
        # Opening JSON file
        j_file = open(txt_dir+j_file)

        # returns JSON object as
        # a dictionary
        j_data = json.load(j_file)

        # Iterating through the json
        # list
        size_obj = len(j_data['boundary_result'])

        mask2 = np.zeros((ydim, xdim), dtype=np.uint8)

        for i in range(0, size_obj):
            mask = np.zeros((ydim, xdim), dtype=np.uint8)
            shape_coord = j_data['boundary_result'][i]
            shape = []
            xpt = int(0)
            ypt = int(1)
            npairs = int((len(shape_coord)-1)/2)
            for uu in range(0, npairs-1):
                xpoint = int(round(shape_coord[xpt]))
                ypoint = round(int(shape_coord[ypt]))
                point = [xpoint, ypoint]
                shape.append(point)
                xpt = xpt+2
                ypt = ypt+2
            contours = np.array(shape)

            # compute the center of the contour

            mask = cv2.fillPoly(mask, np.int32([contours]), 1)
            #mask2 = cv2.fillPoly(mask, np.int32([contours]),1)
            # factor = 0.15

            # height = abs(box_coord[2]-box_coord[6])
            # h_expd = height*(1+factor)
            # h_add = round((abs(height - h_expd)/2))

            # width = abs(box_coord[3]-box_coord[7])
            # w_expd = width*(1+factor)
            # w_add = round((abs(width - w_expd)/2))

            # #draw box as rectangle in mask image
            # cv2.rectangle(mask,(box_coord[2]+h_add,box_coord[3]-w_add),(box_coord[6]-h_add,box_coord[7]+w_add),(1,1,1),-1)
            # #plt.imshow(mask, cmap='plasma')

            #plt.imshow(mask, cmap='plasma')
            # plt.show()
            #mask_inv = cv2.bitwise_not(mask)
            itr_str = str(i).zfill(5)
            mask_inv = np.logical_not(mask)
            mask_inv = np.multiply(mask_inv, 1)
            mask_inv = mask_inv*255
            mask_str_base = os.path.splitext(im_list[ii])[0]
            #mask_str = mask_str_base+"_"+itr_str+"_txt_mask.jpg"
            #maskinv_str = mask_str_base+"_txt_mask_inverted.jpg"
            mask_apld_str = mask_str_base+"_"+itr_str+"_txt_mask_apld.jpg"
            #cv2.imwrite(mask_str, mask)
            #cv2.imwrite(maskinv_str, mask_inv)
            masked_im = (img*mask)+mask_inv
            cv2.imwrite(mask_apld_str, masked_im)
            pause = "pause"


def ind_bbox_txtMask_PANet_IC15(im_dir, txt_dir, im_tag, txt_tag):
    import cv2
    import os
    import glob
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    os.chdir(im_dir)
    # get list of images containing tag
    im_list = glob.glob(im_tag)
    im_list.sort()  # make sure they are sorted in ascending order
    n_im = len(im_list)

    os.chdir(txt_dir)
    # get list of images containing tag
    txt_list = glob.glob(txt_tag)
    txt_list.sort()  # make sure they are sorted in ascending order
    n_txt = len(txt_list)

    os.chdir(im_dir)

    # Get image dimensions
    img4dims = cv2.imread(im_list[0], 0)
    ydim, xdim = img4dims.shape

    for ii in range(0, n_im):
        img = cv2.imread(im_list[ii], 0)
        #mask = np.zeros((ydim,xdim,zdim), np.uint8)
        #mask = np.zeros((ydim,xdim), dtype=np.uint8)

        j_file = txt_list[ii]
        # Opening JSON file
        j_file = open(txt_dir+j_file)

        # returns JSON object as
        # a dictionary
        j_data = json.load(j_file)

        # Iterating through the json
        # list
        size_obj = len(j_data['boundary_result'])

        mask2 = np.zeros((ydim, xdim), dtype=np.uint8)

        for i in range(0, size_obj):
            mask = np.zeros((ydim, xdim), dtype=np.uint8)
            shape_coord = j_data['boundary_result'][i]
            shape = []
            xpt = int(0)
            ypt = int(1)
            npairs = int((len(shape_coord)-1)/2)
            for uu in range(0, npairs-1):
                xpoint = int(round(shape_coord[xpt]))
                ypoint = round(int(shape_coord[ypt]))
                point = [xpoint, ypoint]
                shape.append(point)
                xpt = xpt+2
                ypt = ypt+2
            contours = np.array(shape)

            mask = cv2.fillPoly(mask, np.int32([contours]), 1)
            mask2 = cv2.fillPoly(mask, np.int32([contours]), 1)
            # factor = 0.15

            # height = abs(box_coord[2]-box_coord[6])
            # h_expd = height*(1+factor)
            # h_add = round((abs(height - h_expd)/2))

            # width = abs(box_coord[3]-box_coord[7])
            # w_expd = width*(1+factor)
            # w_add = round((abs(width - w_expd)/2))

            # #draw box as rectangle in mask image
            # cv2.rectangle(mask,(box_coord[2]+h_add,box_coord[3]-w_add),(box_coord[6]-h_add,box_coord[7]+w_add),(1,1,1),-1)
            # #plt.imshow(mask, cmap='plasma')

            #plt.imshow(mask, cmap='plasma')
            # plt.show()
            #mask_inv = cv2.bitwise_not(mask)
            itr_str = str(i).zfill(5)
            mask_inv = np.logical_not(mask)
            mask_inv = np.multiply(mask_inv, 1)
            mask_inv = mask_inv*255
            mask_str_base = os.path.splitext(im_list[ii])[0]
            #mask_str = mask_str_base+"_"+itr_str+"_txt_mask.jpg"
            #maskinv_str = mask_str_base+"_txt_mask_inverted.jpg"
            mask_apld_str = mask_str_base+"_"+itr_str+"_txt_mask_apld.jpg"
            #cv2.imwrite(mask_str, mask)
            #cv2.imwrite(maskinv_str, mask_inv)
            masked_im = (img*mask)+mask_inv
            cv2.imwrite(mask_apld_str, masked_im)
            #pause = "pause"
