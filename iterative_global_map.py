#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 22:00
# @Author  : WYing
# @Site    : 
# @File    : iterative_global_map.py


from PIL import Image
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import copy
from show_image import img_display
from save_image import img_save


def read_json(json_file):
    with open(json_file, 'r') as f:
        load_dict = json.load(f)
    f.close()
    return load_dict

def update_function(global_dict, next_map, ahead_img_name, next_img_name):
    # global_dict -> img_name:cor 
    global_dict_last1 = {key: value[-1] for key, value in global_dict.items()}
    global_img_cor = {value: key for key, value in global_dict_last1.items()}
    # the sorted global coordinates
    global_cor = global_img_cor[ahead_img_name]
    # next_map -> img_name:cor
    next_img_cor = {value: eval(key) for key, value in next_map.items()}
    local_cor = next_img_cor[next_img_name]

    copied_dict = copy.deepcopy(next_img_cor)
    for key, value in next_img_cor.items():
        _value = tuple(a - b for a, b in zip(value, local_cor))
        new_value = tuple(a + b for a, b in zip(_value, global_cor))
        copied_dict[key] = new_value
    copy_next_map = {value: key for key, value in copied_dict.items()}
    for key, value in copy_next_map.items():
        if key in global_dict:
            global_dict[key].append(value)
        else:
            global_dict[key] = [value]

    sorted_dict = dict(sorted(global_dict.items(), key=lambda x: x[0]))
    return sorted_dict

def iterative_function(global_dict, map_paths, pair_paths, iterations):
    for i in range(iterations):
        next_map = read_json(map_paths[i+1])
        # name->cor
        pair_temp = read_json(pair_paths[i])
        ahead_img_name = pair_temp[0]['img_path'][0]
        next_img_name = pair_temp[0]['img_path'][1]
        global_dict = update_function(global_dict, next_map, ahead_img_name, next_img_name)

    return global_dict


if __name__ == '__main__':

    # change the path to your own path
    # ------------------------------------
    os.chdir('./match_results/')
    print(os.getcwd())
    # ------------------------------------

    pair_dir = './image_matches/'   
    json_map_dir = './json_map/'  
    pair_dirs = os.listdir(pair_dir)
    loop_num = len(pair_dirs)
    pair_paths = [os.path.join(pair_dir, name) for name in pair_dirs]

    json_map = os.listdir(json_map_dir)
    map_paths = [os.path.join(json_map_dir, name) for name in json_map]

    pair_temp = read_json(pair_paths[0])
    map_cur_temp = read_json(map_paths[0])   # read the first json file: cor ： img_name
    map_next_temp = read_json(map_paths[1])   

    global_dict = {}
    global_dict = {eval(key): value for key, value in map_cur_temp.items()}
    global_dict = {key: [value] for key, value in global_dict.items()}
 
    result = iterative_function(global_dict, map_paths, pair_paths, loop_num)
    # all images from each location are saved
    img_save(result)
    result_0 = {key: value[0] for key, value in result.items()}
    # img_display(result_0, 'global_result')

    # save the result to json file
    os.makedirs('../figs_global/', exist_ok=True)
    output_file = '../figs_global/global_result.json'
    result_json = {str(key): value for key, value in result.items()}
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(result_json, output_file, ensure_ascii=False, indent=4)

