#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:12 2022

Written by Ethan Duwell, PhD while working a postdoc in the Data Lab at the 
Kern Institute, MCW Milwaukee. 

These functions were written for creating and updating in-house-constructed 
dictionaries for spell-checking/word-checking purposes. In short, 
"pyset_dictionaries" are really just .txt files containing one word per line. 
These are intended to be read in by functions in txt_cleanin_spchk.py, and 
converted to set variables against which lists of words can be checked.

Ethan created these because technical medical terminology/jargon does not tend
to be included in common spell checking software dictionaries and is often flagged
as being mispelled/non-words. The ability to create one's own dictionaries
lets us guard against this by quickly checking words against sets of known 
technical words before flagging any given word as misspelled.

Brief Explanations of Functions in this Module:
    pyset_dictionary: Takes a directory full of pdf &/or .txt files and creates 
                      a .txt dictionary file containing a list of all the 
                      unique words contained in the set of input files.
                      
    
   readin_existing_dict: Reads in existing .txt dictionary files previously
                         created by pyset_dictionary. Returns set variable of
                         all unique words.
                         
   
                         
   dictnry_append: Reads in existing .txt dictionary files previously created 
                   by pyset_dictionary along with a list of additional .pdf or
                   .txt files. Extracts set of unique words from the additional
                   files, adds them to the existing dictionary, and saves a new,
                   updated dictionary containing the additional terms.


For help with spellchecker package visit:
    https://pyspellchecker.readthedocs.io/en/latest/quickstart.html#basic-usage

@author: eduwell
"""

#%% Import Packages
import pdf2txt
import os
import glob
import txt_cleaning_spchk
import copy 
from spellchecker import SpellChecker
import pathstr_chk

#%% Functions
def readin_existing_dict(dict_dir, dict_file):
    # readin_existing_dict reads in existing pyset_dictionaries as a set() of
    # unique words. 
    # Dependencies: txt_cleaning_spchk.getwords() 

    # use txt_cleaning_spchk.getwords() to load dict_file and extract/return set of unique words
    suggest_corrected=3
    words_out, words_out_unique, words_out_unknown,  words_out_unknownWcorr, words_out_known = txt_cleaning_spchk.getwords(dict_dir,dict_file,suggest_corrected,"","",0)
    
    return words_out_unique

def pyset_dictionary(pdf_dir, outfilename):
    
    
    # run pdf2txt on all pdfs in the pdf_dir whose text/words you want to use 
    # for the dictionary (all of the .pdf files in the directory... so only 
    # pdf files in this dir which you want to be included...) 
    pdf2txt.pdf2txt_tag("*.pdf*", pdf_dir, pdf_dir, 0)
    
    
    # Enter the pdf_dir and get a list of the resulting .txt files.
    # Iterate through these to extract the text into a shared list
    os.chdir(pdf_dir)
    tag = "*.txt"
    file_list = glob.glob(tag)
    file_list.sort() # make sure they are sorted in ascending order
    #n_files = len(file_list)
    
    
    suggest_corrected = 3 # set suggest_corrected parameter for txt_cleaning_spchk.getwords
                          # 0: dont suggest corrections, 
                          # 1: suggest the top suggested correction, 
                          # 2: provide list of suggestion candidates.
                          # 3: provide only the lists of known words and unknown words
                          # 4: check unknown words against a selected dictionary .txt file.
                          #    add "unknown" words which match entries in the dictionary into the "known words" set
    
    dict_out = []; # initialize output list
    for txtfiles in file_list:
        words_out, words_out_unique, words_out_unknown,  words_out_unknownWcorr, words_out_known = txt_cleaning_spchk.getwords(pdf_dir,txtfiles,suggest_corrected,"","",0)
        
        words_out_unique = list(words_out_unique) # Convert words_out_unique set to a list
        
        # loop through and append all the words in words_out_unique into dict_out
        for wrds in words_out_unique:
            dict_out.append(wrds)
    
    
    dict_out = set(dict_out) # Convert dict_out to be a set. this will remove duplicates/reduce to minimum set of unique words 
    dict_out = list(dict_out)
    
    # Print the contents of dictionary "dict_out" into out into the output .txt
    # file (outfilename) one word per line
    
    itr8r = 0;
    nwrds = len(dict_out)
    with open(outfilename, 'w') as f:
        for wrds in dict_out:
            f.write(wrds)
            if itr8r < nwrds:
                f.write('\n')
            itr8r = itr8r+1
    f.close()
    del itr8r
    
    return dict_out # return the set dictionary dict_out if specified..
    
def dictnry_append(filedir, filelist, dictFile, apndTag):
    # For appending the text contents of additional .pdf/.txt files into an
    # existing pyset_dictionary
    
    # filedir: full path to directory containing files in filelist
    # filelist: list of files whose text you want to append into the dictionary
    # dictFile: file containing existing pyset_dictionary
    # apndTag: string you want to add to the existing pyset_dictionary name to
    # create unique output file name (ie...dictFile.txt-->"dictFile_apndTag.txt")
    
    # read in existing pyset_dictionary using "readin_existing_dict()"
    existing_dict = readin_existing_dict(filedir, dictFile)
    
    # convert to list for appending stuff..
    existing_dict = list(existing_dict)
    
    # iterate over files in filelist, extract text with txt_cleaning_spchk.getwords()
    # and append to the exiting dictionary list existing_dict
    suggest_corrected = 3
    for txtfiles in filelist:
        file_root = os.path.splitext(txtfiles)[0]
        file_ext = os.path.splitext(txtfiles)[1]
        
        # if file is a pdf.. extract text to .txt file using pdf2txt..
        if file_ext == ".pdf":
            pdf2txt.pdf2txt_tag(txtfiles, filedir, filedir, 0)
            fname = file_root+".txt"
        else:
            #if not.. do nothing but assign txtfiles to var fname for coding 
            # expediency reasons..
            fname = txtfiles
            
        
        words_out, words_out_unique, words_out_unknown,  words_out_unknownWcorr, words_out_known = txt_cleaning_spchk.getwords(filedir,fname,suggest_corrected,"","",0)
        
        words_out_unique = list(words_out_unique) # Convert words_out_unique set to a list
        
        # loop through and append all the words in words_out_unique into dict_out
        for wrds in words_out_unique:
            existing_dict.append(wrds)
    
    existing_dict = set(existing_dict) # Convert dict_out to be a set. this will remove duplicates/reduce to minimum set of unique words
    existing_dict = list(existing_dict)

    dict_file_root = os.path.splitext(dictFile)[0]
    outfilename = dict_file_root+apndTag+".txt"
    
    # write the updated dictionary to file
    itr8r = 0;
    nwrds = len(existing_dict)
    with open(outfilename, 'w') as f:
        for wrds in existing_dict:
            f.write(wrds)
            if itr8r < nwrds:
                f.write('\n')
            itr8r = itr8r+1
    f.close()
    del itr8r
    
    return existing_dict # return the set dictionary dict_out if specified..


def pyset2spellchecker(dict_file, dir_path, mode):
    # Converts an existing pyset_dictionary into a spellchecker dictionary.
    # Note: mode=0 means build the spellchecker dictionary from scratch using
    # only the text in the pyset_dictionary file.
    #
    # mode=1 means add the text in the pyset_dictionary to the existing base
    # english dictionary in spellchecker
    #
    # Another Note: although this was originally built to read in 
    # pyset_dictionary .txt files, dict_file need not be a pyset_dictionary..
    # it can be any .txt file ...
    
    file_str = dir_path+dict_file
    
    if mode == 0:
        # turn off loading a built language dictionary, case sensitive on (if desired)
        psd_spell = SpellChecker(language=None, case_sensitive=False)

        # or... if you have text
        psd_spell.word_frequency.load_text_file(file_str)
    
    if mode == 1:
        psd_spell = SpellChecker()
        psd = readin_existing_dict(dir_path, dict_file)
        psd = list(psd)
        psd_spell.word_frequency.load_words(psd)
    
    # export it out for later use!
    #psd_spell.export('my_custom_dictionary.gz', gzipped=True)
    
    
    return psd_spell

def append2spellchecker(sp_dict,psd):
    # Adds words in an imported pyset_dictionary set to an existing spellcheck 
    # dictionary
    
    psd = list(psd)
    sp_dict.word_frequency.load_words(psd)
    
    return sp_dict

def spchkr_json2dict(file_dir,filename,mode):
    # 
    # mode: 
    #    0: from scratch 
    #    1: load .json on top of base SpellChecker()
    #
    if mode == 0:
        spell = SpellChecker(language=None, case_sensitive=False)
        
    if mode == 1:
        spell = SpellChecker()
    
    # check filename end format.. ensure it ends in a "/" such that it can be 
    # combined with the filename..
    file_dir = pathstr_chk.chk_lastchar(file_dir, 1)
    
    # if you have a dictionary...
    filestr = file_dir+filename
    spell.word_frequency.load_dictionary(filestr)

    return spell

def spchkrparse_txt2dict(file_dir,filename):
    # spellchecker parses existing .txt file and reads into a dictionary 
    spell = SpellChecker()
    
    # check filename end format.. ensure it ends in a "/" such that it can be 
    # combined with the filename..
    file_dir = pathstr_chk.chk_lastchar(file_dir, 1)
    
    filestr = file_dir+filename
    spell.word_frequency.load_text_file(filestr)
    
    return spell

def spchkrApnd_txt2dict(spell, file_dir,filename):
    # uses spellchecker to read in a .txt file and append it to and existing 
    # spellchecker dictionary thats already been loaded in..
    
    # check filename end format.. ensure it ends in a "/" such that it can be 
    # combined with the filename..
    file_dir = pathstr_chk.chk_lastchar(file_dir, 1)
    
    filestr = file_dir+filename
    spell.word_frequency.load_text_file(filestr)
    
    return spell

def spchkr_dict2file(spell,outname,outdir):
    # exports/saves a spellchecker dictionary, 'spell' as a .json file named 
    # 'outname', in directory path location 'outdir'
    # outname example: 'my_custom_dictionary.gz'
    #
    os.chdir(outdir)
    spell.export(outname, gzipped=True)

#%% Test Calls

# #pyset_dictionary test call...
# pdf_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/tst_2"
# outfilename = "pyset_dictnry_webster_netter_medterm.txt"
# test_dictnry = pyset_dictionary(pdf_dir, outfilename)
# pause = ""


# # dictnry_append test call...
# filedir="/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test/pyset_dictionary_test"
# filelist=["Vascular System 2021 - Fritz.pdf"]
# dictFile="test_pyset_dictnry.txt"
# apndTag="_wFritzVasc2021"

# dictnry_append(filedir, filelist, dictFile, apndTag)

# dictnry_append test call...
# filedir="/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test/pyset_dictionary_test"
# filelist=["Vascular System 2021 - Fritz.pdf"]
# dictFile="test_pyset_dictnry.txt"
# apndTag="_wFritzVasc2021"

# dictnry_append(filedir, filelist, dictFile, apndTag)
