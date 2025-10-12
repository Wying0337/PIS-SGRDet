#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 23:10
# @Author  : WYing
# @Site    : 
# @File    : show_image.py

import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl

def img_display(dict_map, file_name):
    img_list = []
    index_list = []
    for key in dict_map:
        key = list(key)
        index_list.append(key)
    # mirror the y axis
    m = max(key[1] for key in dict_map.keys())+1
    for i, cor in enumerate(index_list):
        index_list[i][1] = m - cor[1] - 1

    for value in dict_map.values():
        img_list.append(value)
    img_num = len(index_list)
    height_num = np.max(index_list, 0)[0] + 1
    width_num = np.max(index_list, 0)[1] + 1
    locate = [width_num * each[0] + each[1] + 1 for each in index_list]
    mpl.rcParams['axes.facecolor'] = 'black'
    mpl.rcParams['figure.facecolor'] = 'black'
    title_names = [key for key in dict_map.keys()]
    fig = plt.figure(file_name)
    for i, path in enumerate(img_list):
        img = Image.open(path)
        plt.subplot(height_num, width_num, locate[i])
        plt.imshow(img)
        plt.axis('off')
        plt.title(title_names[i], fontsize=4, color='white', y=0.3)
    plt.subplots_adjust(wspace=0.4, hspace=0.7)   
    plt.savefig(f'../figs_global/{file_name}.jpg', dpi=800)
    # plt.show()