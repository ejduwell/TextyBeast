#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:48:31 2022

@author: eduwell
"""

#%% Import Packages

import os
import pdfplumber
import matplotlib.pyplot as plt

def rctgle_chk(bl, tr, p):  # bl = bottom left (x,y), tr = top right (x,y), p = point coordinate (x,y)
    if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]):
        return True
    else:
        return False

def word_clust(page, words, ClustThr_factor, affinity, linkage):  # ****
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

    

    # Get page dimensions
    ydim = page.height
    xdim = page.width
    
    txt_out_final = []
    
    #make image of page
    img = page.to_image()
    img.draw_rects(words)
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
    #size_obj = len(j_data['box:'])
    
    size_obj = len(words)
    
    
    contour_list = []
    centroid_lst = []
    centroid_lst_wh = []
    clust_heights = []
    text_list = []
    for wrds in words:
        
        UL = [int(round(wrds['top'])),int(round(wrds['x0']))]
        UR = [int(round(wrds['top'])),int(round(wrds['x1']))]
        BL = [int(round(wrds['bottom'])),int(round(wrds['x0']))]
        BR = [int(round(wrds['bottom'])),int(round(wrds['x1']))]
        
        shape = [UL, UR, BL, BR]
        #shape_coord = j_data['result'][i]['box']
        shape_text = wrds['text']
        text_list.append(shape_text)

        contours = np.array(shape)
        contour_xMin = int(round(wrds['x0']))
        
        
        #x_shp, y_shp, w_shp, h_shp = cv2.boundingRect(contours)
        #M = cv2.moments(contours)
        
        w_shp = abs(int(round(wrds['x1']))-int(round(wrds['x0'])))
        h_shp = abs(int(round(wrds['bottom']))-int(round(wrds['top'])))
        cX = abs(int(round(wrds['x1']))+int(round(wrds['x0'])))/2
        cY = abs(int(round(wrds['bottom']))+int(round(wrds['top'])))/2
        
        #EJD COMMENTED BELOW TO TRY USING Xmin (far left instead of the middle to cluster..)
        
        # EJD WENT BACK TO USING CENTROID (FOR PDF WORDS.. bc we're dealing with words instead of lines...)
        centroid = [cX, cY]
        #centroid = [contour_xMin, cY]
        
        centroid_wh = [cX, cY, h_shp]
        clust_heights.append(h_shp)

        centroid_lst.append(centroid)
        centroid_lst_wh.append(centroid_wh)
        contour_list.append(contours) ###
        mask = cv2.fillPoly(mask, np.int32([contours]), 1)


    pause = "pause"


    if len(contour_list) > 1:


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
            if fstpass == 0:
                cl_xyMaxMin = np.vstack(
                    (cl_xyMaxMin, [cx_max, cx_min, cy_max, cy_min]))
            fstpass = 0
            # cl_xyMaxMin.append([cx_max,cx_min,cy_max,cy_min])
        del bb
        del yy

        #cl_labels_cntrds = list(a_set)
        cl_labels_cntrds = list(range(0, n_uVals))

        cluster_mat = np.vstack(
            (copy.deepcopy(cl_labels_cntrds), copy.deepcopy(ave_cluster_centroid_x)))
        cluster_mat = np.vstack(
            (cluster_mat, copy.deepcopy(ave_cluster_centroid_y)))

        cl_xyMaxMin = np.transpose(cl_xyMaxMin)
        cluster_mat = np.vstack((cluster_mat, cl_xyMaxMin))
        
        
        ave_cluster_centroid_y_sort = copy.deepcopy(cluster_mat)
        ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:,ave_cluster_centroid_y_sort[2, :].argsort()] #sort on y
        ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:,ave_cluster_centroid_y_sort[1, :].argsort(kind='mergesort')] #sort on x
        ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:,ave_cluster_centroid_y_sort[2, :].argsort()] #sort on y # (note did this 3 times so things would be organized top->bot left-right)
        
        #EJD COMMENTED BELOW TO REPLACE WITH ABOVE..
        # ave_cluster_centroid_y_sort = np.vstack(
        #     (copy.deepcopy(cl_labels_cntrds), copy.deepcopy(ave_cluster_centroid_y)))
        # ave_cluster_centroid_y_sort = ave_cluster_centroid_y_sort[:,
        #                                                           ave_cluster_centroid_y_sort[1, :].argsort()]
        
        
        
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
        nthr = 0.0 #NOTE EJD TURNED TO 0 to effectively turn off...
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
        # which are the length of the x and y dimensons respectively


# EJD COMMENTED THE "Cut points" section.. didn't look like it was bein used.. 
        # xdim_binTxt = np.zeros((1, xdim), dtype=np.uint8)
        # ydim_binTxt = np.zeros((1, ydim), dtype=np.uint8)

        # for b_boxes in xy_MaxMin:
        #     xrng = list(range(b_boxes[1], b_boxes[0]))
        #     yrng = list(range(b_boxes[3], b_boxes[2]))

        #     if b_boxes[3] > header_ymin:
        #         for nmbrs in xrng:
        #             xdim_binTxt[0][nmbrs] = 1
        #     pause =""
        #     #del nmbrs
        #     if b_boxes[3] > header_ymin:
        #         for nmbrs in yrng:
        #             ydim_binTxt[0][nmbrs] = 1
        #     pause =""
        
        # xdim_gaps_b = []
        # xdim_gaps_e = []
        # ydim_gaps_b = []
        # ydim_gaps_e = []

        # for ww in range(0, xdim_binTxt.shape[1]):

        #     if xdim_binTxt[0][ww] == 0 and ww == 0:
        #         xdim_gaps_b.append(ww)

        #     if xdim_binTxt[0][ww] == 0 and xdim_binTxt[0][ww-1] == 1:
        #         xdim_gaps_b.append(ww)

        #     if ww != (xdim_binTxt.shape[1]-1):
        #         if xdim_binTxt[0][ww] == 0 and xdim_binTxt[0][ww+1] == 1:
        #             xdim_gaps_e.append(ww)

        #     if xdim_binTxt[0][ww] == 0 and ww == (xdim_binTxt.shape[1]-1):
        #         xdim_gaps_e.append(ww)

        # del ww

        # for ww in range(0, ydim_binTxt.shape[1]):

        #     if ydim_binTxt[0][ww] == 0 and ww == 0:
        #         ydim_gaps_b.append(ww)

        #     if ydim_binTxt[0][ww] == 0 and ydim_binTxt[0][ww-1] == 1:
        #         ydim_gaps_b.append(ww)

        #     if ww != (ydim_binTxt.shape[1]-1):
        #         if ydim_binTxt[0][ww] == 0 and ydim_binTxt[0][ww+1] == 1:
        #             ydim_gaps_e.append(ww)

        #     if ydim_binTxt[0][ww] == 0 and ww == (ydim_binTxt.shape[1]-1):
        #         ydim_gaps_e.append(ww)

        # del ww

        # # take the mean of the beginning and end indicies for each gap to
        # # get cutpoints for x and y dimensions..

        # x_cutpnts = []
        # x_cutpnts_siz = []
        # y_cutpnts = []
        # y_cutpnts_siz = []

        # itr8r = 0
        # for ww in xdim_gaps_b:
        #     xcut = ((xdim_gaps_b[itr8r] + xdim_gaps_e[itr8r])/2)
        #     xsiz = (xdim_gaps_e[itr8r]-xdim_gaps_b[itr8r])
        #     x_cutpnts.append(xcut)
        #     x_cutpnts_siz.append(xsiz)
        #     itr8r = itr8r+1
        # x_cutpnts = np.vstack((x_cutpnts, x_cutpnts_siz))
        # del itr8r

        # itr8r = 0
        # for ww in ydim_gaps_b:
        #     ycut = ((ydim_gaps_b[itr8r] + ydim_gaps_e[itr8r])/2)
        #     ysiz = (ydim_gaps_e[itr8r]-ydim_gaps_b[itr8r])
        #     y_cutpnts.append(ycut)
        #     y_cutpnts_siz.append(ysiz)
        #     itr8r = itr8r+1
        # y_cutpnts = np.vstack((y_cutpnts, y_cutpnts_siz))
        # del itr8r

        # Iterate through clusters below the header looking for horizontal
        # conflicts. If one is found, add cut nearest to the rightmost edge
        # of the rightmost b_box in conflict. Rinse and repeat.. (iterating)
        # through each segment chunk individually untill no conflicts arise
# EJD COMMENTED THE "Cut points" section.. didn't look like it was bein used..

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

        pause = "pause"  #!!! 0-1 missing in main lists as of here...!!!
        
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
                        header_lsts_nodups.append(tmp_lst)
                        fpass = 0;
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
            
            # if not os.path.exists("cluster_viz"):
            #     os.mkdir("cluster_viz")
            # os.chdir("cluster_viz")
            
            # try: header_lsts
            # except NameError: header_lsts = None            

            # if header_lsts is None:
            #     do_nothing = "nothing"
            # else:
            #     arb_it = 0
            #     for ee in range(0,(header_lst_cntrds.shape[1])):
            #         clstr_o_clstrs = header_lsts_nodups_coords_srt[header_lst_cntrds[0][ee]]
            #         for qq in range(0,(clstr_o_clstrs.shape[1])):
            #             clst = clstr_o_clstrs[0][qq]
            #             mask2 = np.zeros((ydim, xdim), dtype=np.uint8)
            #             clist = clusters[round(clst)]
    
            #             for bb in clist:
            #                 mask2 = cv2.fillPoly(mask2, np.int32([bb]), 1)
            #             del bb
            #             mask2_inv = np.logical_not(mask2)
            #             mask2_inv = np.multiply(mask2_inv, 1)
            #             mask2_inv = mask2_inv*255
            #             mask2_str_base = os.path.splitext(im_list[ii])[0]
            #             mask2_str = mask2_str_base+"_" + \
            #                 str(arb_it).zfill(3)+"_txt_mask_apld.jpg"
            #             masked_im2 = (img*mask2)+mask2_inv
            #             masked_im_0b = (img*mask2)
    
            #             cv2.imwrite(mask2_str, masked_im2)
    
            #             arb_it = arb_it+1
            
            
            # if header_lsts is None:
            #     arb_it = 0   
            # else:
            #     do_nothing = "nothing"
            
            # try: main_lsts
            # except NameError: main_lsts = None
            
            # if main_lsts is None:
            #     do_nothing = "nothing"
            # else: 
            #     for ee in range(0,(main_lst_cntrds.shape[1])):
            #         clstr_o_clstrs = main_lsts_nodups_coords_srt[main_lst_cntrds[0][ee]]
            #         for qq in range(0,(clstr_o_clstrs.shape[1])):
            #             clst = clstr_o_clstrs[0][qq]
            #             mask2 = np.zeros((ydim, xdim), dtype=np.uint8)
            #             clist = clusters[round(clst)]
    
            #             for bb in clist:
            #                 mask2 = cv2.fillPoly(mask2, np.int32([bb]), 1)
            #             del bb
            #             mask2_inv = np.logical_not(mask2)
            #             mask2_inv = np.multiply(mask2_inv, 1)
            #             mask2_inv = mask2_inv*255
            #             mask2_str_base = os.path.splitext(im_list[ii])[0]
            #             mask2_str = mask2_str_base+"_" + \
            #                 str(arb_it).zfill(3)+"_txt_mask_apld.jpg"
            #             masked_im2 = (img*mask2)+mask2_inv
            #             masked_im_0b = (img*mask2)
    
            #             cv2.imwrite(mask2_str, masked_im2)
    
            #             arb_it = arb_it+1
 
            # pause = "pause"
            # os.chdir(im_dir)
            ##################################################################
            ##################################################################
    return txt_out_final


def pdf2txt_wrd_clust(pdf_filename, pdf_dir, txt_outname, txt_outdir, ClustThr_factor, affinity, linkage, x_tol, y_tol):
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
        
        txtout = [];
        itr8r = 0;
        for pgs in pages:
            words = pgs.extract_words(x_tolerance=x_tol, y_tolerance=3, keep_blank_chars=False, use_text_flow=False, horizontal_ltr=True, vertical_ttb=True, extra_attrs=[])
            #txt = pgs.extract_text(x_tolerance=3, y_tolerance=3, layout=True, x_density=7.25, y_density=13)
            pagetxt = word_clust(pgs, words, ClustThr_factor, affinity, linkage)
            txtout.append(pagetxt)
            
    
    frame_itr = 0;
    with open(txt_outname, 'w') as f:
        for frames in txtout:
            frame_str = "TEXT SCRAPED FROM PAGE #"+" "+str(frame_itr).zfill(4)
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
            
            cntr = 0
            for txt_clusters in frames:
                for sub_clustrs in txt_clusters:
                    if cntr > 0:
                        f.write('\n')
                    for txt_strgs in sub_clustrs:
                        for ind_strgs in txt_strgs:
                            f.write(ind_strgs)
                            f.write('\n')
                            #f.write(' ') # EJD TRYING SPACE INSTEAD..
                f.write('\n')
                f.write('\n')
                cntr = cntr+1
            frame_itr = frame_itr+1
    f.close()
        
#%% Test Calls

pdf_filename = "Vascular System 2021 - Fritz.pdf"
pdf_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/" 
txt_outname = "Vascular System 2021 - Fritz.txt"
txt_outdir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/PDF_Text_Extraction/"
ClustThr_factor = 10; # controls agglomerative clustering threshold distance (in units of n*average textbox height)
affinity='euclidean'
linkage = 'ward'
x_tol=3
y_tol=3

pdf2txt_wrd_clust(pdf_filename, pdf_dir, txt_outname, txt_outdir, ClustThr_factor, affinity, linkage, x_tol, y_tol)
pause = ""
                
                
                