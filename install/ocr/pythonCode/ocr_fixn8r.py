#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:20:40 2022

Written by Ethan Duwell, PhD.

ocr_fixn8r is a module containing functions for identifying and correcting
common errors in ocr detetion/recognition performed by MMOCR functions in
frm2txt_DL2.py and lecxr_text.py


@author: eduwell
"""


#%% Import Packages
from spellchecker import SpellChecker
import pyset_dictionary
import os
#%% Functions

def missing_space_dtxn(dict_file, dict_filedir, wrds2chk):
    # ID and correct instances where two words get stuck together w/out a space
    #
    # Parameters:
        # dict_file: 
        #    option 1: string, custom dictionary .txt file name
        #    option 2: Spellchecker object, an existing spellchecker dictionary
        # dict_filedir: string, full path to custom dictionary .txt file directory
        # wrds2chk: list, list of words to check
    #
    # How does it work/what does it do?:
    # for each string in the wrds2chk list:
        # cut the string at positions 1->(n-1), each time checking whether the
        # two pieces are both a word.. This will effectively try every possible
        # location at which a space might have been skipped. For words in which
        # >0 cut position results in 2 viable words which checkout against the
        # dictionary, include the original string and the potential resulting 
        # string-word chunks in a sub-list within the output list.
    
    # setup the spellchceker dictionary
    if dict_file != "":
        check = 0;
        if repr(type(dict_file)) == "<class 'spellchecker.spellchecker.SpellChecker'>":
            spell = dict_file # do nothing.. its already a spellcheck dictionary..
            check = 1;
        
        if repr(type(dict_file)) == "<class 'str'>":
            # its a filename for a saved dictionary not a spellcheck dictionary
            # check the filetype extension..
            f_ext = os.path.splitext(dict_file)[1]
            check = 1;
            # load into spellchecker based on detected filetype
            if f_ext == ".txt":
                spell = pyset_dictionary.pyset2spellchecker(dict_file, dict_filedir, 1) # create spellcheck dictionary with custom pyset_dictionary terms added
            if f_ext == ".json":
                spell = pyset_dictionary.spchkr_json2dict(dict_filedir,dict_file, 1)
        if check == 0:
            spell = SpellChecker() # if no file specified, use spellchecker's base english dictionary instead..
    
    #initialize output list
    missing_space_wrds = [];
    
    # Iterate over words in wrds2chk: For each word, try all possible cut
    # positions checking whether the resulting chunks are viable words on each
    # cut position. If >0 cut positions result in a pair of viable words, 
    # include them in the final output..
    
    for wrds in wrds2chk:
        pos_cuts = []; # list to store original string (idx 0) and viable/possible cut chunks (lists of 2 strings) which produce words (idxs 1:n)
        pos_cuts.append(wrds) # add orig string to idx 0
        nchar = len(wrds);
        for num in range(1,(nchar-1)):
            chnk1 = wrds[0:num];
            chnk2 = wrds[num:nchar];
            
            chnks_temp = [chnk1, chnk2] # save chunks in temp list to check against dictionary

            known = spell.known(chnks_temp) # check against dictionary 
           
            # if both chunks matched add the word to the pos_cuts list
            if len(known)>1:
                pos_cuts.append(chnks_temp)
        
        # check whether there was >=1 case where cutting the string results in 2 viable words
        # if so, add pos_cuts to missing_space_wrds
        if len(pos_cuts)>1:  # note: >1 because the first index in the list is always the orig string.. ****!
            missing_space_wrds.append(pos_cuts)
            
    return missing_space_wrds
    
    
def stickychar_dtxn(dict_in,wrds2chk):
    # ID and correct instances of "sticky characters".. ie Ethan's observation of
    # instances where last letter of a word gets stuck/copied
    # onto the beginning of the next word and similar instances in which the
    # first character of a word gets stuck on the end of the previous word..
    
    if dict_in == None:
        spell = SpellChecker() # initialize SpellChecker as an object
    else:
        spell = dict_in
    
    #initialize output list
    stickychar_wrds = [];
    
    for wrds in wrds2chk:
        
        # get the chunks containing strings with characters [0->(N-1)], [1->N], and [1->(N-1)]
        # (these are effectively the string with the last letter chopped off,
        # the string with the first letter chopped off and the string with both
        # the first and last letter chopped off...)
        
        nchar = len(wrds);
        pos_cuts = [wrds,[],[],[]]; # list to store original string (idx 0) and viable/possible cut chunks which produce words (idxs 1:n)
        
        chnk1=wrds[0:(nchar-1)]
        chnk2=wrds[1:nchar]
        chnk3=wrds[1:(nchar-1)]
        
        chnks_temp = [chnk1, chnk2, chnk3] # save chunks in temp list to check against dictionary

        # if at least one of the chunks matches as a word, add the original 
        # string/the viable chunks to the missing_space_words list.
        match_cntr = 0 # initialize match counter
        for cc in range(0,len(chnks_temp)):
            chunks = chnks_temp[cc]
            #d_chk = dict_in.intersection([chunks]) # check against the in-house pyset_dictionary (dict_in)
            known = spell.known([chunks]) # check against built-in dictionary in spellchecker.SpellChecker()
                
            if len(known)>0:
                pos_cuts[cc+1] = chunks
        
        # check whether there was >=1 case where cutting the string results in a viable word
        # if so, add pos_cuts to stickychar_wrds
        if len(pos_cuts[1])>0:
            stickychar_wrds.append(pos_cuts)
            
    return stickychar_wrds
    
def simlr_charswap_dtxn(wrds2chk):
    # ID and correct cases in which geometrically similar/confusable chararacters
    # are mistaken for one another (ie.. c mistaken for e, etc..)
    pholder =""
    
#%% Test Calls           

# # -------tst call for missing_space_dtxn-------
# tst_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test/pyset_dictionary_test/tst_dir"

# phony_list = ["components","word","componentsword"];
# tst_dict_file = "test_pyset_dictnry.txt"

# import pyset_dictionary as psd

# tst_dict = psd.readin_existing_dict(tst_dir, tst_dict_file) # read in the test dictionary

# ms_wrds_tst = missing_space_dtxn(tst_dict, phony_list)

# pause ="";


# # -------tst call for stickychar_dtxn-------
# tst_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test/pyset_dictionary_test/tst_dir"

# phony_list = ["components","word","componentsword", "zwordz","xword","wordx"];
# tst_dict_file = "test_pyset_dictnry.txt"

# import pyset_dictionary as psd

# tst_dict = psd.readin_existing_dict(tst_dir, tst_dict_file) # read in the test dictionary

# stky_wrds_tst = stickychar_dtxn(tst_dict, phony_list)

# pause ="";



            
            
                