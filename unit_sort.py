#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   uint_sort.py
@Time    :   2025/07/07 00:29:34
@Author  :    WYing
@Version :   1.0
@Desc    :   None
'''

import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

def read_json(json_path):
    """
    Read a JSON file and return its content.
    
    :param json_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data  


def get_array(unit_path, json_path):
    """
    Read the JSON file and extract coordinates.
    
    :param unit_path: Path to the unit directory.
    :param json_path: Path to the JSON file.
    :param cor: List to store coordinates.
    :return: Tuple of result coordinates and dictionary name.
    """
    cor = []  # coordinate
    full_path = [] # save the image path corresponding to coordinate
    json_data = read_json(json_path)
    for i, item in enumerate(json_data):
        x1 = item['box']['x1']
        y1 = item['box']['y1']
        x1 = round(x1, 2)
        y1 = round(y1, 2)
        path = item['img_path']  # obtain the path of each unit
        cor.append((x1, y1))
        each_unit_path = os.path.join(unit_path, path.split('/')[-1])
        full_path.append(each_unit_path)
    unit_path_dict = {k: v for k, v in zip(cor, full_path)}  # path 
    return cor, unit_path_dict

def sort_array(result_cor, flag=0):
    """
    Sort the coordinates first by x and then by y.
    
    :param result_cor: List of coordinates.
    :return: Sorted indices of the coordinates.
    """
    list_xy = [list(item) for item in result_cor]
    array_xy = np.array(list_xy)
    if flag == 0:
        # Sort by the X-axis first, then by the Y-axis
        index = np.lexsort((array_xy[:, 1], array_xy[:, 0]))  # The index is used to align with the unit.
    else:
        # Sort by the Y-axis first, then by the X-axis
        index = np.lexsort((array_xy[:, 0], array_xy[:, 1]))
    x_sorted_cor = array_xy[index]
    flag = int(flag)
    residual = [0]
    for i in range(len(x_sorted_cor) - 1):
        # Calculate the difference between adjacent coordinates
        residual.append(x_sorted_cor[i+1, :][flag] - x_sorted_cor[i, :][flag])
    residual = np.array(residual)
    index_peak_value = np.where(residual > 300)[0]
    index_peak_value = np.append(index_peak_value, len(x_sorted_cor))
    cols = []   
    cols.append(index_peak_value[0])
    for i in range(len(index_peak_value)-1):
        cols.append(index_peak_value[i+1] - index_peak_value[i])
    col_num = len(cols)
    index_col = []
    for i, ind_num in enumerate(cols):
        temp_index = [i]*ind_num
        index_col.extend(temp_index)
    for j, index in enumerate(index_col):
        index_col[j] = col_num-index-1     # invert the coordinate
    x_sorted_cor_tuple = [tuple(item) for item in x_sorted_cor]
    xy_col_dict = {k: v for k, v in zip(x_sorted_cor_tuple, index_col)}  
    return xy_col_dict

def img_display(dict_xy_path, img_dir):
    file_name = img_dir
    img_list = []
    index_list = []
    for key in dict_xy_path:
        key = list(key)
        index_list.append(key)

    # Mirror the dimension y axis
    m = max(key[1] for key in dict_xy_path.keys()) + 1
    for i, cor in enumerate(index_list):
        index_list[i][1] = m - cor[1] - 1
      
    for value in dict_xy_path.values():
        img_list.append(value)
    img_num = len(index_list)
    height_num = np.max(index_list, 0)[0] + 1
    width_num = np.max(index_list, 0)[1] + 1
    locate = [width_num * each[0] + each[1] + 1 for each in index_list]

    plt.figure(file_name)
    # set the image size
    plt.figure(figsize=(width_num * 2, height_num * 2))  # image size
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.1, hspace=0.1)  # Adjust the subplot spacing

    # plt.axis('off')  # axis off
    for i, path in enumerate(img_list):
        img = Image.open(path)
        plt.subplot(height_num, width_num, locate[i])
        plt.imshow(img)
        plt.axis('off')
        # plt.title('image')
    plt.savefig(f'./figs/{file_name}.jpg')
    plt.show()

if __name__ == "__main__":
    # change the work directory
    os.chdir('./detect_results')
    # json file and uint directory for sorting
    json_dir = './jsons'
    unit_dir = './unit'
    # Iterate through each JSON file
    img_dirs= os.listdir(unit_dir)
    img_dirs.sort()
    
    for img_dir in img_dirs:
        unit_path = os.path.join(unit_dir, img_dir)
        # Read the JSON file
        json_path = os.path.join(json_dir, img_dir + '.json')

        # Get the coordinates and unit paths
        result_cor, unit_path_dict = get_array(unit_path, json_path)   
        # Sort the coordinates x and y
        dict_col_index = sort_array(result_cor, flag=0)
        # Sort the coordinates y and x
        dict_row_index = sort_array(result_cor, flag=1)
        tuple_xy = [tuple(xy) for xy in result_cor]
        dict_xy_index = {k: (dict_row_index[k], dict_col_index[k]) for k in tuple_xy}
        # Create a mapping from coordinates to unit names
        dict_xy_path = {dict_xy_index[key]: unit_path_dict[key] for key in dict_xy_index}
        # Mirror the first dimension (x-axis) of the coordinates in dict_xy_path
        max_x = max(coord[0] for coord in dict_xy_path.keys())
        dict_xy_path = {(max_x - coord[0], coord[1]): path for coord, path in dict_xy_path.items()}
        print(f'dict_xy_path: {dict_xy_path}')

       
        
        os.makedirs('./json_local_map/', exist_ok=True)
        os.makedirs('./figs', exist_ok=True)
        with open(f'./json_local_map/{img_dir}.json', 'w') as f:
            dict_temp = {str(k): dict_xy_path[k] for k in dict_xy_path}
            sorted_dict = {key: dict_temp[key] for key in sorted(dict_temp)}
            json.dump(sorted_dict, f, indent=4, ensure_ascii=False)
        print('the json file has been saved.')

        img_display(dict_xy_path, img_dir)
       
    

     


        


