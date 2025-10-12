#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 20:40
# @Author  : WYing
# @Site    : 
# @File    : save_image.py

import os
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import shutil

def img_save(result):
    for coordinates, image_paths in result.items():
        folder_name = f"{coordinates[0]}_{coordinates[1]}"
        folder_dir = os.path.join('../figs_global', folder_name)

        # make directory
        os.makedirs(folder_dir, exist_ok=True)

        # copy images to the directory
        for image_path in image_paths:
            shutil.copy(image_path, folder_dir)