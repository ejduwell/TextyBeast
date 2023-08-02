#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:46:03 2022

@author: eduwell
"""

### import the word tokenizer and stop words from nltk, along with other modules that are required
import re
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import os
from spellchecker import SpellChecker
import dict_chkr
import copy
import pyset_dictionary

def getwords(directory,txtin,suggest_corrected,dict_file,dict_filedir,mode):
    # Parameter Notes:
    # suggest_corrected = _   # 0: dont suggest corrections, 
                              # 1: suggest the top suggested correction, 
                              # 2: provide list of suggestion candidates.
                              # 3: provide only the lists of known words and unknown words
                              # 4: check unknown words against a selected pyset_dictionary .txt file using dictionary_chkr.
                              #    add "unknown" words which match entries in the dictionary into the "known words" set
                              # 5: check unknown words against a selected pyset_dictionary .txt file using 
                              #    combined with spellchecker english dictionary. Suggest the top suggested correction
                              # 6: check unknown words against a selected pyset_dictionary .txt file using 
                              #    combined with spellchecker english dictionary. Provides list of suggestion candidates.
                              # 7: use existing .json dictionary (ie such as the as the guttenberg one from: https://github.com/dwyl/english-words)
                              #    Suggest the top suggested correction
                              # 8: use existing .json dictionary (ie such as the as the guttenberg one from: https://github.com/dwyl/english-words)
                              #    Provide list of suggestion candidates.
                              # 9: same as 7 but .json is loaded on top of base spellchecker() dictionary..
                              # 10: same as 8 but .json is loaded on top of base spellchecker() dictionary..
                              # 11: same as 9 but also adds an additional custom .txt file in too
                              # 12: same as 10 but also adds an additional custom .txt file in too
                              
    # For "mode" parameter: mode=0 is for .txt files, as "txtin", mode2 is for lists as "txtin"
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')
    
    if mode == 0: #(input is a text file)
        ### Read the text file
        #directory ="/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/data/Course_Materials_Data/CHA/CHA1-2_VIDEO/test_trimmed/out_dir/video_img_dta/unique_frames/mmocr_out" 
        os.chdir(directory)
        #txtin = 'tst_trimmed2_framesText.txt'
        with open(txtin) as f:
            text = f.read()
        
    if mode == 1: # (input txtin is simply a list of words .. instead of a file..)
        text = repr(txtin);
    
    ### Here we will regenerate tokens from our original text. We'll call them tokens
    tokens_list = word_tokenize(text)
    print("\n" + "First 50 tokens: " + "\n" + str(tokens_list[:50]))
    orig_num_words = len(tokens_list) # count the original number of words in the text.
    
    ##########################
    ### Insert our filtering for caretted words: 
    caret_words = list() # make a blank list that will hold our caret words.
    
    # Iterate through the entire list of words. If a '^' exists, add it to our 'caret_words' list.
    for i in tokens_list:
        a = re.search('[A-Z0-9]',i) # regular expression search for any capital letter or numeral
        b = re.search('[\^]',i)     # regular expression to search for a caret ('^') symbol
        if a and b:                 # An if statement that is true if both regular expressions find a match
            caret_words.append(i)   # if true, add to the list of caretted words to be removed
            tokens_list.remove(i)         # if true, remove the matching words from our list
    
            # Print the list: 
    print("Caretted words removed : " + "\n" + str(list(caret_words)))
    
    
    
    
    ### convert to lower case
    tokens_list = [i.lower() for i in tokens_list]
    
    ##########################
    ### Remove punctuation by making a replacment table between pucntuation and empty spaces
    repl_table = str.maketrans('', '', string.punctuation)
    tokens_nopunct = [i.translate(repl_table) for i in tokens_list]
    print("\n" + "First 50 words after removing punctuation: " + "\n" + str(tokens_nopunct[:50]))
    
    ##############################
    # Get rid of any other non-alphanumeric characters
    words = [word for word in tokens_nopunct if word.isalpha()]
    
    print("\n" + "First 50 words after removing non-alphanumeric: " + "\n" + str(words[:50]))
    
    ##############################
    # Remove stopwords with nltk
    words = [i for i in words if not i in stop_words]
    print("\n" + "First 50 words after removing stop words: " + "\n" + str(words[:50]))
    
    # Print an update on number of words remaining
    print("Original number of words: " + str(orig_num_words) + "; Number remaining: " + str(len(words)))
    
    unknown_sugg = [[""]]
    unknown = []
    if suggest_corrected > 0:
        
        # set up spellchecker based on specified input parameters..
        if suggest_corrected == 1 or suggest_corrected == 2:
            spell = SpellChecker() # initialize SpellChecker as an object
        
        if suggest_corrected == 5 or suggest_corrected == 6:
            spell = pyset_dictionary.pyset2spellchecker(dict_file, dict_filedir, 1)

        if suggest_corrected == 7 or suggest_corrected == 8:
            spell = pyset_dictionary.spchkr_json2dict(dict_filedir,dict_file,0) 
            
        if suggest_corrected == 9 or suggest_corrected == 10:
            spell = pyset_dictionary.spchkr_json2dict(dict_filedir,dict_file,1)
            
        if suggest_corrected == 11 or suggest_corrected == 12:
            # read in .json (first index) first
            spell = pyset_dictionary.spchkr_json2dict(dict_filedir[0],dict_file[0],1)
            # now add the .txt file (second index)
            spell = pyset_dictionary.spchkrApnd_txt2dict(spell, dict_filedir[1],dict_file[1])
        
        # run SpellChecker ALL items in our list of words and return only misspelled
        unknown = spell.unknown(words[:])
        known = spell.known(words[:])
        # print the list of mispelled items from the first 100 words:
        print("misspelled words: " + str(list(unknown)))
        
        if suggest_corrected == 1 or suggest_corrected == 2 or suggest_corrected == 5 or suggest_corrected == 6 or suggest_corrected == 7 or suggest_corrected == 8 or suggest_corrected == 9 or suggest_corrected == 10 or suggest_corrected == 11 or suggest_corrected == 12:
            unknown_sugg = []
            for i in unknown:
                #print("original: " + i + "; suggested: " + spell.correction(i))
                orig = i
                
                if (suggest_corrected == 1) or (suggest_corrected == 5) or suggest_corrected == 7 or suggest_corrected == 9 or suggest_corrected == 11:
                    suggested= spell.correction(i)
                    
                if suggest_corrected == 2 or suggest_corrected == 6 or suggest_corrected == 8 or suggest_corrected == 10 or suggest_corrected == 12:
                    suggested= spell.candidates(i)
                    
                    
                pair = [orig,suggested]
                unknown_sugg.append(pair)
                    
        if suggest_corrected == 4:
            list2chk = list(unknown)
            dict_matches = dict_chkr.dictionary_chkr(dict_file,dict_filedir,list2chk)
                    
            dict_matches = list(dict_matches)
            known_tmp = copy.deepcopy(known)
            known_tmp = list(known_tmp)

            unknown_tmp = copy.deepcopy(unknown)
            unknown_tmp = list(unknown_tmp)                    
                    
                    
            for nn in dict_matches:
                known_tmp.append(nn)
                unknown_tmp.remove(nn)
                    
            known = set(known_tmp)
            unknown = set(unknown_tmp)
    
    words_unique = set(words)

    return words, words_unique, unknown, unknown_sugg, known, spell


def fnd_rplc_ignorecase(txtfile_in, txtfile_out, dir_in, dir_out, wrd_lst,export):
    
    os.chdir(dir_in)
    
    # opening the txtfile_in in read mode
    file = open(txtfile_in, "r")
    replacement = ""
    # using the for loop
    fst_pass = 1;
    for jj in wrd_lst:
        orig_str = jj[0] #!!
        replmt_str =  jj[1] #!!
        if fst_pass == 1:
            for line in file:
                line = line.strip()
                replacer = re.compile(re.escape(orig_str), re.IGNORECASE) # Note: would be better to move this up just below replmt_str = jj[1]
                
                fxd_line = replacer.sub(r'\b'+replmt_str+'\b', line)
                
                
                #changes = line.replace(jj[0], jj[1])
                replacement = replacement + fxd_line + "\n"
        else:
            replacer = re.compile(re.escape(orig_str), re.IGNORECASE)
            
            replacement = replacer.sub(r'\b'+replmt_str+'\b', replacement)
            
            
            pause = ""
                
        fst_pass = 0;
    file.close()
    
    
    os.chdir(dir_out)
    
    if export == 1:
        # opening the file in write mode
        fout = open(txtfile_out, "w")
        fout.write(replacement)
        fout.close()

    return replacement

def fnd_rplc_prtReplacement(replacement,txtfile_out, dir_out):

    os.chdir(dir_out)

    fout = open(txtfile_out, "w")
    fout.write(replacement)
    fout.close()

#TEST CALL
# directory ="/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test" 
# txtin = 'Podcast Guide - Anterior Cranial Fossa.txt'
# suggest_corrected = 4 # 0: dont suggest corrections, 
#                       # 1: suggest the top suggested correction, 
#                       # 2: provide list of suggestion candidates.
#                       # 3: provide only the lists of known words and unknown words
#                       # 4: check unknown words against a selected dictionary .txt file.
#                       #    add "unknown" words which match entries in the dictionary into the "known words" set
                      
# dict_file = "Podcast Guide - Anterior Cranial Fossa.txt"
# dict_filedir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/Texts_for_Dictionaries/test"

# words_out, words_out_unknown,  words_out_unknownWcorr, known_words = getwords(directory,txtin,suggest_corrected,dict_file,dict_filedir,0)

# pause = ""
