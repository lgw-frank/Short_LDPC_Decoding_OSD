# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:34:56 2021

@author: Administrator
"""
import os
import globalmap as GL
def display_selection():
    print("My piD: " + str(os.getpid()))
    if GL.get_map('selected_decoder_type') == 'SPA':
        print("Using Sum-Product Algorithm")
    else:
        print("Using one of Min-Sum variants")
    