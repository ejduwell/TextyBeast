#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:14:59 2022
#Params:
# path_str: string, full path to a directory..
# end_format: desired "format" for the end of the string.
    # end_format = 0: last character is NOT a '/'
    # end_format = 1: last character is a '/'
# What this function does:
    # Checks the last character of a directory path string and checks whether
    # it matches the desired end_format specified.
    #
    # If it does not match, the end of the path string is modified such that
    # the output path_str does..

# get last character in the string
@author: eduwell
"""

def chk_lastchar(path_str,end_format):
    #Params:
    # path_str: string, full path to a directory..
    # end_format: desired "format" for the end of the string.
        # end_format = 0: last character is NOT a '/'
        # end_format = 1: last character is a '/'
    # What this function does:
        # Checks the last character of a directory path string and checks whether
        # it matches the desired end_format specified.
        #
        # If it does not match, the end of the path string is modified such that
        # the output path_str does..
    
    # get last character in the string
    last_char = path_str[-1]
    
    
    if end_format == 0:
        if last_char == "/":
            path_str = path_str[0:-1]
    
    if end_format == 1:
        if last_char != "/":
            path_str = path_str+"/"
    
    return path_str