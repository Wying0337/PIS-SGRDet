#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 21:57
# @Author  : WYing
# @Site    : 
# @File    : optimize_patch_for_each_cor.py


from PIL import Image
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')
from show_image import img_display

def read_json(json_file):
    with open(json_file, 'r') as f:
        load_dict = json.load(f)
    f.close()
    return load_dict

def get_h_w(coordinate_dict):
    max_h = max(value[0] for value in coordinate_dict.values())
    max_w = max(value[1] for value in coordinate_dict.values())
    center_h = np.floor(max_h/2)
    center_w = np.floor(max_w/2)
    return max_h, max_w

def merge_json_files(input_folder, output_file):
    all_data = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            json_dict = {value: eval(key) for key, value in json_data.items()}
            center_h, center_w = get_h_w(json_dict)
            json_dict = {key: (value + (center_h, center_w,)) for key, value in json_dict.items()}
            all_data.append(json_dict)

    # combine all json data
    merged_data = {}
    for data in all_data:
        merged_data.update(data)

    # save to new file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=2)


def merge_json2(dict1, dict2):
    # dict1 = {
    #     "(0, 1)": [
    #         "./dish\\2023_09_23_16_01_02_087_950_0_0_0\\2023_09_23_16_01_02_087_950_0_0_03.jpg",
    #         "./dish\\2023_09_23_16_01_12_071_950_179_0_20\\2023_09_23_16_01_12_071_950_179_0_206.jpg",
    #         "./dish\\2023_09_23_16_03_52_235_1150_59_0_20\\2023_09_23_16_03_52_235_1150_59_0_2010.jpg"
    #     ]
    # }
    #
    # dict2 = {
    #     "./dish\\2023_09_23_16_01_02_087_950_0_0_0\\2023_09_23_16_01_02_087_950_0_0_03.jpg": "(0, 1)"
    # }
    dict_3_merge = {}
    for key1, values1 in dict1.items():
        nested_dict = {v: key1 for v in values1}
        for key_n in list(nested_dict.keys()):
            for key2, value2 in dict2.items():
                if key2 == key_n:
                    nested_dict[key2] = value2
        # store the nested mapping once for this key1
        dict_3_merge[key1] = [nested_dict]
    output_file = '../figs_global/dict_3_merge.json'
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(dict_3_merge, output_file, ensure_ascii=False, indent=2)
    return dict_3_merge

def select_dict(dict_3_merge):
    dict_4_select = {}
    for key, values in dict_3_merge.items():
        ratio = []
        flag_value = []
        names = []
        for name, cor in values[0].items():
            local_cor_x, local_cor_y, cor_h, cor_w = cor
            image_cur = Image.open(name)
            img_h, img_w = image_cur.size
            ratio_t = img_h/img_w if img_h < img_w else img_w/img_h
            ratio.append(ratio_t)
            flag_value_t = (abs(local_cor_x + 1 - (cor_h + 1) / 2)/(cor_h + 1)
                             + abs(local_cor_y + 1 - (cor_w + 1) / 2))/(cor_w + 1)
            flag_value.append(flag_value_t)
            names.append(name)
        ratio_and_value = list(zip(names, ratio, flag_value))
        print(f'ratio_and_value: {ratio_and_value}')
        sorted_ratio = sorted(ratio_and_value, key=lambda x: x[1], reverse=True)
        sorted_value = sorted(ratio_and_value, key=lambda x: x[2])
        result = []
        for item in sorted_value:
            if item[1] > 0.90:
                result.append(item[0])
        dict_4_select.update({key: result[0]})
    result = {eval(key): value for key, value in dict_4_select.items()}
    return dict_4_select, result


if __name__ == '__main__':
    # ------------------------------------
    os.chdir('./match_results/')
    print(os.getcwd())
    # ------------------------------------

    json_map_dir = './json_map/'  
    output_file = '../figs_global/merged.json'

    merge_json_files(json_map_dir, output_file)
    # combined json file
    dict_1 = read_json('../figs_global/global_result.json')
    dict_2_merge = read_json('../figs_global/merged.json')
    dict_3_merge = merge_json2(dict_1, dict_2_merge)
    # josn file and image display
    dict_4_select, result_selected = select_dict(dict_3_merge)
    img_display(result_selected, 'final_global_result_b')
    with open(f'../figs_global/result_selected.json', 'w') as f:
        f.write(json.dumps(dict_4_select, indent=4))
    f.close





